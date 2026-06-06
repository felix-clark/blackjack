use crate::card::Card;
use std::{
    fmt::{Debug, Display},
    ops::{Add, Sub},
};

/// Number of distinct card ranks: Ace, 2..=9, Ten.
pub const N_RANKS: usize = 10;

/// A multiset of cards stored densely as a per-rank tally indexed by [`Card::rank_index`] — the
/// single representation for both a hand and a shoe.
///
/// Blackjack only ever distinguishes ten ranks, so a fixed array is the natural backing store:
/// equality and hashing are exact and cheap (an absent rank is simply a `0`, with none of the
/// "explicit zero vs. missing key" ambiguity a `HashMap`-backed multiset has), and the whole thing
/// is `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CardCol {
    counts: [u16; N_RANKS],
}

impl CardCol {
    pub fn new() -> Self {
        Self {
            counts: [0; N_RANKS],
        }
    }

    pub fn len(&self) -> usize {
        self.counts.iter().map(|&n| n as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&n| n == 0)
    }

    pub fn get_count(&self, card: &Card) -> u16 {
        self.counts[card.rank_index()]
    }

    /// Add one card of the given rank.
    pub fn insert(&mut self, card: Card) {
        self.counts[card.rank_index()] += 1;
    }

    /// Add `n` cards of the given rank.
    pub fn add_n(&mut self, card: Card, n: u16) {
        self.counts[card.rank_index()] += n;
    }

    /// Remove every card of the given rank, returning how many there were.
    pub fn remove_rank(&mut self, card: Card) -> u16 {
        let index = card.rank_index();
        let n = self.counts[index];
        self.counts[index] = 0;
        n
    }

    /// The highest-valued rank present, if any.
    pub fn highest_rank(&self) -> Option<Card> {
        self.counts
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &n)| n > 0)
            .map(|(index, _)| Card::from_rank_index(index))
    }

    /// Iterate the ranks that are present, paired with their counts.
    pub fn iter(&self) -> impl '_ + Iterator<Item = (Card, u16)> {
        self.counts
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(index, &n)| (Card::from_rank_index(index), n))
    }

    /// True if every rank's count is `<=` the corresponding count in `other`, i.e. `self` could be
    /// drawn from `other`.
    pub fn is_submultiset(&self, other: &Self) -> bool {
        self.counts
            .iter()
            .zip(&other.counts)
            .all(|(mine, theirs)| mine <= theirs)
    }

    pub fn from_decks(n: u8) -> Self {
        let mut counts = [0; N_RANKS];
        let n_per_rank = 4 * n as u16;
        for i in 2..=9 {
            counts[Card::Pip(i).rank_index()] = n_per_rank;
        }
        counts[Card::Ten.rank_index()] = 4 * n_per_rank;
        counts[Card::Ace.rank_index()] = n_per_rank;
        Self { counts }
    }

    #[allow(unused)]
    pub fn half_deck() -> Self {
        let mut counts = [0; N_RANKS];
        let n_per_rank = 2;
        for i in 2..=9 {
            counts[Card::Pip(i).rank_index()] = n_per_rank;
        }
        counts[Card::Ten.rank_index()] = 4 * n_per_rank;
        counts[Card::Ace.rank_index()] = n_per_rank;
        Self { counts }
    }

    pub fn from_hand(hand: &[Card]) -> Self {
        let mut col = Self::new();
        for &card in hand {
            col.insert(card);
        }
        col
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
        self.counts
            .iter()
            .enumerate()
            .map(|(index, &n)| (index as u8 + 1) * n as u8)
            .sum()
    }

    pub fn has_ace(&self) -> bool {
        self.counts[Card::Ace.rank_index()] > 0
    }

    /// A natural blackjack: one Ace (rank index 0) and one Ten (rank index `N_RANKS - 1`).
    const NAT21: Self = {
        let mut counts = [0u16; N_RANKS];
        counts[0] = 1;
        counts[N_RANKS - 1] = 1;
        Self { counts }
    };

    pub fn is_nat21(&self) -> bool {
        *self == Self::NAT21
    }

    /// Lazily enumerate every multiset of cards drawn from this deck whose hard total (aces low)
    /// equals `hard_total`, each paired with its multivariate-hypergeometric scan-weight. Seeds the
    /// traversal with `norm_offset` at 0 so callers never supply it; returns a [`WeightedPartitions`]
    /// [`Iterator`] of `(weight, hand)` pairs.
    pub fn weighted_partitions(&self, hard_total: u8) -> WeightedPartitions {
        WeightedPartitions {
            stack: vec![PartitionFrame {
                deck: *self,
                hard_total,
                norm_offset: 0,
                hand: CardCol::new(),
                weight: 1.0,
            }],
        }
    }
}

/// One pending recursive "call" in the lazy partition enumeration: a sub-deck still to be
/// enumerated against a remaining hard total, plus the prefix of cards already chosen by ancestor
/// frames and the scan-weight accumulated down to this point.
struct PartitionFrame {
    /// Remaining ranks strictly below the ones already branched on (the sub-deck).
    deck: CardCol,
    /// Hard total still to be filled by this frame and its descendants.
    hard_total: u8,
    /// Normalization bookkeeping threaded into this level's scan-weight.
    norm_offset: u16,
    /// Cards chosen by ancestor frames; a leaf emits this hand.
    hand: CardCol,
    /// Product of ancestors' scan-weights. The telescoped `weight_part` (== `N!`, the factorial of
    /// the final hand size) is applied once at the leaf, not here.
    weight: f64,
}

/// Lazy, allocation-light partition enumerator produced by [`CardCol::weighted_partitions`].
///
/// This is an explicit-stack depth-first traversal of the rank-branching tree the recursive
/// `_weighted_partitions_legacy` reference walks. It yields the same `(weight, hand)` pairs without
/// materializing a `Vec` at every node — only one reusable stack `Vec` is kept alive.
///
/// The weights are computed via the telescoping identity for the per-level `weight_part` factors:
/// across all levels of a partition they collapse to `N!`, where `N` is the total hand size. So a
/// leaf's weight is `N!` times the running product of per-level scan-weights, the latter
/// accumulated cheaply on the way down. This is algebraically identical to the recursive version
/// (modulo floating-point reassociation well within the `~1e-10` cross-checks).
pub struct WeightedPartitions {
    stack: Vec<PartitionFrame>,
}

impl Iterator for WeightedPartitions {
    type Item = (f64, CardCol);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(frame) = self.stack.pop() {
            let PartitionFrame {
                mut deck,
                hard_total,
                norm_offset,
                hand,
                weight,
            } = frame;

            // Leaf: the remaining total is filled. The product of every level's `weight_part`
            // telescopes to `N!`, so apply it here in one shot.
            if hard_total == 0 {
                let n = hand.len();
                let n_fact = (1..=n).map(|k| k as f64).product::<f64>();
                return Some((weight * n_fact, hand));
            }
            // Ran out of cards before reaching the total: dead end, emit nothing.
            if deck.is_empty() {
                continue;
            }

            let n_deck = deck.len() as f64;
            // Highest remaining rank; removing it leaves exactly the sub-deck of lower ranks.
            let top_rank: Card = deck.highest_rank().expect("deck is non-empty");
            let n_top = deck.remove_rank(top_rank);
            let sub_deck = deck;

            // Push a child frame for each count `k_top` of `top_rank`, stopping once `top_rank`
            // alone overshoots the target (monotonic in `k_top`, matching the original `break`).
            //
            // NOTE: We push k_top = 0, 1, 2, ... and pop LIFO, so children come out in reverse
            // k-order relative to the recursive version. Order is irrelevant to the real consumer
            // (`build_hard_evs` keys a HashMap); reverse the loop if a matching printout is wanted.
            let mut level_weight = 1.0;
            for k_top in 0..=n_top {
                let top_cont = top_rank.hard() * k_top as u8;
                if top_cont > hard_total {
                    break;
                }

                let mut child_hand = hand;
                child_hand.add_n(top_rank, k_top);

                self.stack.push(PartitionFrame {
                    deck: sub_deck,
                    hard_total: hard_total - top_cont,
                    norm_offset: norm_offset + n_top - k_top,
                    hand: child_hand,
                    weight: weight * level_weight,
                });

                // Advance the scan-weight for the next `k_top`. This is
                //     (n_top CHOOSE k_top) / [(n_deck + norm_offset) falling-factorial k_top],
                // built incrementally exactly as in the recursive version.
                let k = k_top as f64;
                level_weight *= n_top as f64 - k;
                level_weight /= k + 1.;
                level_weight /= n_deck + norm_offset as f64 - k;
            }
        }
        None
    }
}

impl Add for CardCol {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (mine, theirs) in self.counts.iter_mut().zip(&rhs.counts) {
            *mine += theirs;
        }
        self
    }
}

impl Sub for CardCol {
    type Output = Self;

    /// Per-rank saturating subtraction (counts never go below zero), matching the multiset
    /// semantics the solver relies on.
    fn sub(mut self, rhs: Self) -> Self::Output {
        for (mine, theirs) in self.counts.iter_mut().zip(&rhs.counts) {
            *mine = mine.saturating_sub(*theirs);
        }
        self
    }
}

impl TryFrom<&str> for CardCol {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut col = Self::new();
        for ch in value.chars() {
            col.insert(Card::try_from(ch)?);
        }
        Ok(col)
    }
}

impl Display for CardCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (card, num) in self.iter() {
            write!(f, "{}:{} ", card, num)?;
        }
        Ok(())
    }
}

impl Debug for CardCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (card, num) in self.iter() {
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
        self.counts[card.rank_index()] -= 1;
    }

    fn draw_prob(&self, card: &Card) -> f64 {
        let denom = self.len() as f64;
        self.counts[card.rank_index()] as f64 / denom
    }

    fn all_draw_probs(&self) -> impl Iterator<Item = (Card, f64)> {
        let denom = self.len() as f64;
        self.iter().map(move |(card, n)| (card, n as f64 / denom))
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
