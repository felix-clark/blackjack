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

    /// Get the count using the index i, rather than the Card. This is used in some other
    /// algorithms, like the count conditioning, that also utilize the array representation for data
    /// over card ranks.
    pub fn get_count_i(&self, i: usize) -> u16 {
        self.counts[i]
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
}

/// One pending recursive "call" in the lazy partition enumeration: the highest rank still to be
/// branched on, the hard total still to fill, the cards already chosen by ancestor frames, and the
/// scan-weight accumulated down to this point.
struct PartitionFrame {
    /// Highest rank index still to branch on; ranks are processed high → low, and `None` means none
    /// remain. (A leaf is detected by `hard_total == 0`, independent of this.)
    next_rank: Option<usize>,
    /// Hard total still to be filled by this frame and its descendants.
    hard_total: u8,
    /// Cards chosen by ancestor frames; a leaf emits this hand.
    hand: CardCol,
    /// Product of ancestors' scan-weights. The telescoped per-level factor (== `N!`, the factorial
    /// of the final hand size) is applied once at the leaf, not here.
    weight: f64,
    /// Hypergeometric bookkeeping: cards left in the (finite) shoe after the draws already chosen —
    /// the running falling-factorial denominator. Unused for the multinomial (infinite) law.
    remaining: u16,
}

/// Lazy, allocation-light partition enumerator produced by [`Shoe::weighted_partitions`].
///
/// This is an explicit-stack depth-first traversal of the rank-branching tree the recursive
/// `_weighted_partitions_legacy` reference walks. It yields the same `(weight, hand)` pairs without
/// materializing a `Vec` at every node — only one reusable stack `Vec` is kept alive.
///
/// The weights are computed via the telescoping identity for the per-level scan-weight factors:
/// across all levels of a partition they collapse to `N!`, where `N` is the total hand size. So a
/// leaf's weight is `N!` times the running product of per-level factors, the latter accumulated
/// cheaply on the way down.
///
/// The per-level factor is chosen by the shoe, per rank, via [`Shoe::rank_count`]:
/// - `Some(n)` (finite shoe): multivariate hypergeometric, `C(n, k) / fallingfactorial(remaining, k)`,
///   telescoping across levels to `∏_r C(n_r, k_r) / C(N_deck, N)` — drawing without replacement.
/// - `None` (infinite deck): multinomial, `p_rank^k / k!` with `p_rank` read live from
///   [`Shoe::draw_prob`] (a constant, since the deck doesn't deplete), so the leaf yields
///   `N! · ∏_r p_r^{k_r}/k_r!` — drawing with replacement.
pub struct WeightedPartitions<S: Shoe> {
    stack: Vec<PartitionFrame>,
    shoe: S,
}

impl<S: Shoe> Iterator for WeightedPartitions<S> {
    type Item = (f64, CardCol);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(frame) = self.stack.pop() {
            let PartitionFrame {
                next_rank,
                hard_total,
                hand,
                weight,
                remaining,
            } = frame;

            // Leaf: the remaining total is filled. The product of every level's factor telescopes
            // to `N!`, so apply it here in one shot.
            if hard_total == 0 {
                let n = hand.len();
                let n_fact = (1..=n).map(|k| k as f64).product::<f64>();
                return Some((weight * n_fact, hand));
            }
            // No ranks left but the total is still unmet: dead end, emit nothing.
            let Some(rank) = next_rank else { continue };

            let top_rank = Card::from_rank_index(rank);
            let value = top_rank.hard() as u16;
            // The next-lower rank to branch on in every child frame.
            let child_rank = rank.checked_sub(1);
            // Most copies of `top_rank` that still fit under the target (further bounded by the
            // shoe's finite supply, below).
            let max_k = hard_total as u16 / value;

            // Push a child frame for each count `k` of `top_rank`, advancing the running per-level
            // scan-weight `level_weight` as we go. We push k = 0, 1, 2, ... and pop LIFO, so children
            // come out in reverse k-order; order is irrelevant to the consumer (`build_evs`
            // keys a HashMap).
            let mut level_weight = 1.0;
            match self.shoe.rank_count(&top_rank) {
                // Finite shoe: hypergeometric. `n_top` is this rank's count; the factor advances by
                // `(n_top - k)/(k+1)` (the C(n_top, k) ratio) over the falling-factorial term of the
                // running `remaining` count. Telescopes to the hypergeometric PMF.
                Some(n_top) => {
                    for k in 0..=max_k.min(n_top) {
                        let mut child_hand = hand;
                        child_hand.add_n(top_rank, k);
                        self.stack.push(PartitionFrame {
                            next_rank: child_rank,
                            hard_total: hard_total - (k * value) as u8,
                            hand: child_hand,
                            weight: weight * level_weight,
                            remaining: remaining - k,
                        });
                        let kf = k as f64;
                        level_weight *= (n_top - k) as f64;
                        level_weight /= kf + 1.;
                        level_weight /= remaining as f64 - kf;
                    }
                }
                // Infinite deck: multinomial. `p_rank` is constant (no depletion), so the factor just
                // advances by `p_rank/(k+1)` to build `p_rank^k / k!`.
                None => {
                    let p = self.shoe.draw_prob(&top_rank);
                    for k in 0..=max_k {
                        let mut child_hand = hand;
                        child_hand.add_n(top_rank, k);
                        self.stack.push(PartitionFrame {
                            next_rank: child_rank,
                            hard_total: hard_total - (k * value) as u8,
                            hand: child_hand,
                            weight: weight * level_weight,
                            remaining,
                        });
                        level_weight *= p;
                        level_weight /= k as f64 + 1.;
                    }
                }
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

    /// The shoe remaining after a whole hand (a multiset of cards) is removed: multiset difference
    /// for a finite shoe, unchanged for a non-depleting one.
    fn remove_hand(&self, hand: &CardCol) -> Self;

    /// Whether `hand` could be drawn from this shoe — always true for a non-depleting shoe.
    fn contains_hand(&self, hand: &CardCol) -> bool;

    /// How many of `rank` the shoe holds: `Some(count)` for a finite shoe, `None` for a
    /// non-depleting one (the infinite deck). This is the partition enumerator's only per-rank input:
    /// it bounds how many copies a hand may take and selects the scan-weight law — `Some` →
    /// hypergeometric (drawing without replacement), `None` → multinomial (with replacement, drawing
    /// probabilities read from [`Shoe::draw_prob`]).
    fn rank_count(&self, rank: &Card) -> Option<u16>;

    /// Lazily enumerate every multiset of cards drawn from this shoe whose hard total (aces low)
    /// equals `hard_total`, each paired with its scan-weight (see [`WeightedPartitions`]). Returns a
    /// [`WeightedPartitions`] [`Iterator`] of `(weight, hand)` pairs.
    fn weighted_partitions(&self, hard_total: u8) -> WeightedPartitions<Self>
    where
        Self: Sized,
    {
        // Total finite supply seeds the hypergeometric falling-factorial denominator; for an
        // infinite deck every `rank_count` is `None`, leaving this 0 and unused.
        let remaining = (0..N_RANKS)
            .filter_map(|i| self.rank_count(&Card::from_rank_index(i)))
            .sum();
        WeightedPartitions {
            stack: vec![PartitionFrame {
                next_rank: Some(N_RANKS - 1),
                hard_total,
                hand: CardCol::new(),
                weight: 1.0,
                remaining,
            }],
            shoe: self.clone(),
        }
    }
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

    fn remove_hand(&self, hand: &CardCol) -> Self {
        *self - *hand
    }

    fn contains_hand(&self, hand: &CardCol) -> bool {
        hand.is_submultiset(self)
    }

    /// A finite shoe holds a concrete count of each rank (and so draws without replacement →
    /// hypergeometric weights).
    fn rank_count(&self, rank: &Card) -> Option<u16> {
        Some(self.get_count(rank))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
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

    /// The infinite deck never depletes, so removing a hand leaves it unchanged and it can supply
    /// any hand.
    fn remove_hand(&self, _hand: &CardCol) -> Self {
        *self
    }

    fn contains_hand(&self, _hand: &CardCol) -> bool {
        true
    }

    /// The infinite deck has unbounded copies of every rank — `None` — which both leaves the
    /// partition `k` bound to the hard total alone and selects multinomial (with-replacement)
    /// weights, with probabilities read live from [`InfiniteDeck::draw_prob`].
    fn rank_count(&self, _rank: &Card) -> Option<u16> {
        None
    }
}
