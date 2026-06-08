//! The hand/move/strategy vocabulary: the collapsed [`HandState`] a concrete hand maps to, the
//! player [`Move`] options, and the strategy-table [`HandCategory`] a hand is charted under, plus the
//! [`pair_rank`]/[`categorize`] helpers that route a concrete [`CardCol`] into these. The exact-hand
//! EV engine lives in [`crate::simulation`]; this is the abstract layer the charts are built over.

use std::fmt::Display;

use crate::card::Card;
use crate::shoe::CardCol;

#[derive(PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum HandState {
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
pub(crate) enum Move {
    Hit,
    Stand,
    Double,
    Split,
    Surrender,
}

// TODO: similar to Move, we might need an enum for recommended strategy, which encodes DoubleHit
// and DoubleStand (to double if allowed, but Hit/Stand otherwise).

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

/// The row a concrete hand occupies in a strategy table.
///
/// Distinct from [`HandState`]: a pair is *also* a hard or soft total, but it is a different
/// decision (split is available, and only here), so it gets its own category rather than being
/// pooled into the corresponding total. `A,A` is `Pair(Ace)` and `T,T` is `Pair(Ten)` — neither
/// falls through to `Soft`/`Hard`/`Natural`. Hard and soft categories still pool every composition
/// (and size) of that total, which is where composition-dependent strategy is averaged out.
#[derive(PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum HandCategory {
    Hard(u8),
    Soft(u8),
    Pair(Card),
    Natural,
}

impl Display for HandCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandCategory::Hard(n) => write!(f, "H{}", n),
            HandCategory::Soft(n) => write!(f, "S{}", n),
            HandCategory::Pair(c) => write!(f, "{},{}", c, c),
            HandCategory::Natural => write!(f, "Nat"),
        }
    }
}

/// The rank of a two-card pair, if the hand is one (exactly two cards of the same rank).
pub(crate) fn pair_rank(hand: &CardCol) -> Option<Card> {
    if hand.len() != 2 {
        return None;
    }
    hand.iter().find(|&(_, n)| n == 2).map(|(c, _)| c)
}

/// Route a concrete hand to its strategy-table row (see [`HandCategory`]). Pairs take priority over
/// the hard/soft total they also form; everything else defers to [`HandState`].
pub(crate) fn categorize(hand: &CardCol) -> HandCategory {
    if let Some(rank) = pair_rank(hand) {
        return HandCategory::Pair(rank);
    }
    match HandState::from(hand) {
        HandState::Natural => HandCategory::Natural,
        HandState::Soft(n) => HandCategory::Soft(n),
        HandState::Hard(n) => HandCategory::Hard(n),
        // The tree only holds hands totalling at most 21, so a stored hand is never bust.
        HandState::Bust => unreachable!("a stored hand totals at most 21, so it is never bust"),
    }
}
