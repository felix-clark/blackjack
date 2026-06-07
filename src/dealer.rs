use crate::card::Card;
use crate::shoe::{CardCol, Shoe};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum DealerOutcome {
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

/// Number of distinct dealer outcomes: Bust, Total(17..=21), Natural.
const N_DEALER_OUTCOMES: usize = 7;

/// A probability distribution over dealer outcomes, laid out densely:
/// index 0 = Bust, indices 1..=5 = Total(17..=21), index 6 = Natural.
type DealerDist = [f64; N_DEALER_OUTCOMES];

/// Index of a dealer outcome in a [`DealerDist`].
fn dealer_outcome_index(outcome: &DealerOutcome) -> usize {
    match outcome {
        DealerOutcome::Bust => 0,
        DealerOutcome::Total(n) => (n - 16) as usize,
        DealerOutcome::Natural => 6,
    }
}

/// Inflate a dense [`DealerDist`] back into the sparse `HashMap` the callers expect.
fn dealer_dist_to_map(dist: DealerDist) -> HashMap<DealerOutcome, f64> {
    let mut out = HashMap::new();
    if dist[0] > 0. {
        out.insert(DealerOutcome::Bust, dist[0]);
    }
    for total in 17..=21u8 {
        let p = dist[dealer_outcome_index(&DealerOutcome::Total(total))];
        if p > 0. {
            out.insert(DealerOutcome::Total(total), p);
        }
    }
    if dist[6] > 0. {
        out.insert(DealerOutcome::Natural, dist[6]);
    }
    out
}

/// The dealer's cards as a dense per-rank tally (indexed by [`Card::rank_index`]).
///
/// This is the dealer's entire state — every hit/stand decision and the final total are derived
/// from it — and it is also the memoization key. Keying on the tally rather than the draw sequence
/// is what collapses the factorial of draw orders: `5` then `6` and `6` then `5` reach the same
/// `DealerHand`, so the subtree below it is solved once.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DealerHand {
    counts: [u8; 10],
}

impl DealerHand {
    fn from_cards(cards: &CardCol) -> Self {
        let mut counts = [0u8; 10];
        for (card, n) in cards.iter() {
            counts[card.rank_index()] = n as u8;
        }
        Self { counts }
    }

    /// The hand with one more card of the rank at `index`.
    fn with_card(mut self, index: usize) -> Self {
        self.counts[index] += 1;
        self
    }

    fn num_cards(&self) -> u32 {
        self.counts.iter().map(|&n| n as u32).sum()
    }

    fn has_ace(&self) -> bool {
        self.counts[0] > 0
    }

    /// Total with every ace counted as 1.
    fn hard_total(&self) -> u32 {
        self.counts
            .iter()
            .enumerate()
            .map(|(index, &n)| (index as u32 + 1) * n as u32)
            .sum()
    }

    /// Best total not exceeding 21, promoting a single ace to 11 when it fits.
    fn best_total(&self) -> u32 {
        let hard = self.hard_total();
        if self.has_ace() && hard <= 11 {
            hard + 10
        } else {
            hard
        }
    }

    /// A natural is the only two-card 21: ace + ten.
    fn is_natural(&self) -> bool {
        self.num_cards() == 2 && self.best_total() == 21
    }

    /// Whether the dealer must take another card, given the soft-17 rule.
    fn must_hit(&self, hs17: bool) -> bool {
        let hard = self.hard_total();
        if hard >= 17 {
            return false;
        }
        // A soft total stands at 18, or at 17 too unless the house hits soft 17.
        let soft_stand_total = if hs17 { 18 } else { 17 };
        let stands_soft = self.has_ace() && hard <= 11 && hard + 10 >= soft_stand_total;
        !stands_soft
    }

    /// The outcome once the dealer stops drawing.
    fn terminal_outcome(&self) -> DealerOutcome {
        let total = self.best_total();
        if total > 21 {
            DealerOutcome::Bust
        } else if self.is_natural() {
            DealerOutcome::Natural
        } else {
            DealerOutcome::Total(total as u8)
        }
    }
}

/// Exact, *unconditional* distribution over dealer outcomes from a given starting hand and shoe —
/// the dealer natural is left in as its own outcome (mass `P(nat)`), never stripped here.
///
/// This is deliberately rule-agnostic apart from `hs17`. The dealer-natural (peek) conditioning is
/// **not** applied in this hot path: a natural is a flat `-1` on every player line of three-plus
/// cards, so conditioning on "no dealer natural" is an affine transform with the same constants for
/// every move and only changes *reported* EVs, not the argmax. It therefore belongs once, at the
/// 2-card root of the EV tree (see `build_evs`), not threaded through this recursion. Callers that
/// genuinely want the conditioned distribution can apply [`remove_nat21`] themselves.
///
/// Generic over any [`Shoe`], so it serves both a finite [`CardCol`] (draws deplete the tally) and
/// the [`InfiniteDeck`](crate::shoe::InfiniteDeck) (draws are no-ops at fixed 1/13 probabilities)
/// through one code path. The recursion is memoized on the [`DealerHand`]: for a fixed starting shoe
/// the remaining shoe is a function of the cards the dealer has drawn, so the dealer hand alone is a
/// sound key, and this collapses the factorial of draw orders to the distinct reachable hands.
pub fn dealer_outcome_probs(
    hand: CardCol,
    shoe: &impl Shoe,
    hs17: bool,
) -> HashMap<DealerOutcome, f64> {
    let mut memo: HashMap<DealerHand, DealerDist> = HashMap::new();
    let dist = dealer_dist(DealerHand::from_cards(&hand), shoe, hs17, &mut memo);
    dealer_dist_to_map(dist)
}

fn dealer_dist<S: Shoe>(
    hand: DealerHand,
    shoe: &S,
    hs17: bool,
    memo: &mut HashMap<DealerHand, DealerDist>,
) -> DealerDist {
    if !hand.must_hit(hs17) {
        let mut dist = [0.0; N_DEALER_OUTCOMES];
        dist[dealer_outcome_index(&hand.terminal_outcome())] = 1.0;
        return dist;
    }
    if let Some(&dist) = memo.get(&hand) {
        return dist;
    }

    // Average the sub-distributions of each possible next card, weighted by its draw probability.
    let mut dist = [0.0; N_DEALER_OUTCOMES];
    for rank in 0..10 {
        let card = Card::from_rank_index(rank);
        let prob = shoe.draw_prob(&card);
        if prob == 0.0 {
            continue;
        }
        // Branch on drawing `card`: clone the shoe and deplete it (a no-op for the infinite deck).
        let mut sub_shoe = shoe.clone();
        sub_shoe.draw(&card);
        let sub = dealer_dist(hand.with_card(rank), &sub_shoe, hs17, memo);
        for (acc, p) in dist.iter_mut().zip(sub) {
            *acc += prob * p;
        }
    }
    memo.insert(hand, dist);
    dist
}

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
