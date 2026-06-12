use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Debug, Display};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Serialize, Deserialize)]
pub enum Card {
    Ace,
    /// a.k.a. Numeral cards, 2-9
    Pip(u8),
    /// In blackjack, tens and face cards are all equivalent.
    Ten,
}

impl Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Card::Pip(n) => write!(f, "{}", n),
            Card::Ten => write!(f, "T"),
            Card::Ace => write!(f, "A"),
        }
    }
}
impl Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    /// Dense-array index for this rank: Ace -> 0, Pip(n) -> n - 1, Ten -> 9.
    pub fn rank_index(&self) -> usize {
        (self.hard() - 1) as usize
    }

    /// Inverse of [`Card::rank_index`].
    pub fn from_rank_index(index: usize) -> Self {
        Card::try_from(index as u8 + 1).expect("valid rank index")
    }
}
