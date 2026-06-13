//! The chart's solve configuration: the shoe choice and the optional card-counting condition, plus the
//! option lists the rules modal cycles through. [`ShoeChoice::solve`] is the per-column entry point the
//! worker threads call (disk-cached on its full key).

use serde::{Deserialize, Serialize};

use crate::card::Card;
use crate::count::{CountCmp, CountShoe, CountSystem, Ko, Penetration};
use crate::diskcache;
use crate::rules::Ruleset;
use crate::shoe::{CardCol, InfiniteDeck, Shoe};
use crate::simulation::insurance_ev;

use super::column::{Column, solve_counted, solve_on};

/// Penetration prior used for count conditioning: a flat distribution over deck depth up to 75%
/// penetration (casinos never deal the shoe out). See the count-conditioning architecture notes.
pub(super) const COUNT_PENETRATION: Penetration = Penetration::FlatPastPercent(25);

/// A card-counting condition the chart is solved under: a counting system (KO for now), the player's
/// external running count, and how it is compared. `None` of this is applied on the infinite deck (an
/// infinite deck has no count) or when counting is toggled off.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct CountSetting {
    /// The player's external running count value being conditioned on.
    pub(super) external: i16,
    /// How the running count is compared to `external` (`==`, `≥`, `≤`).
    pub(super) cmp: CountCmp,
}

impl CountSetting {
    pub(super) fn cmp_label(self) -> &'static str {
        match self.cmp {
            CountCmp::Eq => "==",
            CountCmp::Ge => ">=",
            CountCmp::Le => "<=",
        }
    }
}

/// The shoe the chart is solved against: an infinite (non-depleting) deck or a finite `n`-deck shoe.
/// This is the seam a future card-counting input would adjust.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) enum ShoeChoice {
    Infinite,
    Decks(u8),
}

impl std::fmt::Display for ShoeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShoeChoice::Infinite => write!(f, "\u{221e}"),
            ShoeChoice::Decks(n) => write!(f, "{n}"),
        }
    }
}

impl ShoeChoice {
    /// Solve one up-card's full EV tree on this shoe, collapsing it to the per-category strategy
    /// summary the chart renders and the two-card-root [`EdgeTerm`] the footer's overall edge sums.
    /// Both are read off the same tree, so the edge costs no extra solve. Runs on a worker thread.
    /// `count` conditions the solve on a card-counting running count; it only applies to a finite shoe
    /// (an infinite deck has no count) and is ignored when `None`.
    ///
    /// [`EdgeTerm`]: crate::simulation::EdgeTerm
    pub(super) fn solve(
        self,
        up_card: Card,
        rules: &Ruleset,
        count: Option<CountSetting>,
    ) -> Column {
        // Disk cache: a solved column is fully determined by (up-card, shoe, ruleset, count condition),
        // so persist it — a revisited configuration loads instantly instead of re-solving (splits and
        // all). Best-effort; a miss/error just recomputes.
        let key = (up_card, self, *rules, count);
        if let Some(col) = diskcache::load::<_, Column>("column", &key) {
            return col;
        }
        let column = match self {
            ShoeChoice::Infinite => solve_on(InfiniteDeck {}, up_card, rules),
            ShoeChoice::Decks(n) => match count {
                Some(c) => solve_counted(n, c.external, c.cmp, up_card, rules),
                None => solve_on(CardCol::from_decks(n), up_card, rules),
            },
        };
        diskcache::store("column", &key, &column);
        column
    }

    /// Return the insurance expectation value for the current count state. This is a 2:1 bet that
    /// the dealer has a natural, and is essentially independent of the player's hand outside of the
    /// count implications.
    /// `ShoeChoice` is a UI *selection*, so its one job here is to dispatch to the concrete shoe the
    /// player faces; the actual EV is [`insurance_ev`] (in the solver, beside `dealer_natural_prob`).
    /// The branch is irreducible — the three arms are distinct concrete `Shoe` types and `Shoe` is not
    /// object-safe — but each arm is now just a constructor, with the draw-then-evaluate shared.
    pub(super) fn insurance(self, count: Option<CountSetting>) -> f64 {
        let up = Card::Ace;
        match (self, count) {
            // The entered count includes the dealer's up-card (the player has seen the Ace and counted
            // it before deciding insurance — the Wizard-of-Odds convention). So we anchor the count at
            // the *pre-up-card* value `external - map(Ace)`; `insurance_after_up`'s all-shift `draw`
            // then shifts it back by `map(Ace)`, landing the hole-card distribution at exactly
            // `external` with the Ace removed. Without this offset the hole card would be conditioned on
            // `external + map(Ace)` instead.
            (ShoeChoice::Decks(n), Some(c)) => insurance_after_up(
                CountShoe::from_external::<Ko>(n, c.external - Ko::map(&up), c.cmp, COUNT_PENETRATION),
                up,
            ),
            (ShoeChoice::Decks(n), None) => insurance_after_up(CardCol::from_decks(n), up),
            (ShoeChoice::Infinite, _) => insurance_after_up(InfiniteDeck {}, up),
        }
    }
}

/// Remove the up-card from `shoe` (a no-op on the infinite deck; multiset subtraction on a finite or
/// count shoe), then evaluate insurance against the resulting hole-card distribution. The seam that
/// lets [`ShoeChoice::insurance`]'s arms stay one-constructor-each.
fn insurance_after_up<S: Shoe>(mut shoe: S, up: Card) -> f64 {
    shoe.draw(&up);
    insurance_ev(up, &shoe)
}

/// Deck options the rules modal cycles through.
pub(super) const DECK_OPTIONS: [ShoeChoice; 6] = [
    ShoeChoice::Infinite,
    ShoeChoice::Decks(1),
    ShoeChoice::Decks(2),
    ShoeChoice::Decks(4),
    ShoeChoice::Decks(6),
    ShoeChoice::Decks(8),
];

/// Split-precision options the rules modal cycles through (`split_cards` budget). The fully exact
/// cross-arm search (a budget larger than any reachable draw count) is intentionally not offered — it
/// is combinatorially infeasible on a big shoe and only used in tests.
pub(super) const SPLIT_OPTIONS: [u8; 7] = [0, 1, 2, 3, 4, 6, 8];
