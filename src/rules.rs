//! Rule configuration: the [`Ruleset`] knobs (and the [`PeekRule`] peek+surrender axis) that
//! parametrise the solver. Pure data — no compute lives here. CLAUDE.md anticipates these becoming
//! first-class and threaded through the dealer/payoff logic in place of hardcoded defaults.

use serde::{Deserialize, Serialize};

/// The dealer's hole-card peek and the player's surrender option, as a *single* axis because they are
/// not independent: *late* surrender is defined as surrendering after a clean peek, so it cannot exist
/// in a no-peek game. Bundling them lets the type system enforce that — the invalid (no-peek, late)
/// combination is simply unrepresentable, so the solver needs no runtime check to reject it.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum PeekRule {
    /// The dealer peeks at the hole card before the player acts, so a dealer natural takes only the
    /// original bet (doubled and split bets are returned). Surrender, if offered, may be early or late.
    Peek(PeekSurrender),
    /// European no-hole-card: the dealer draws the hole card only after the player finishes, so a
    /// late-revealed natural takes doubled and split bets too. Late surrender is incoherent here (there
    /// is no "after the peek"), so the only choice is whether *early* surrender is offered.
    NoPeek { early_surrender: bool },
}

/// When (if ever) the player may forfeit half the bet in a peek game.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum PeekSurrender {
    /// Surrender is not offered.
    None,
    /// Surrender *before* the dealer peeks, escaping the dealer-natural loss too. Unconditional -0.5.
    Early,
    /// Surrender *after* the dealer peeks and shows no blackjack. Conditional -0.5.
    Late,
}

impl PeekRule {
    /// Whether the dealer peeks at the hole card. This is the bit that changes the EV basis: off peek,
    /// doubled and split bets are also forfeited to a dealer natural revealed at the end.
    pub(crate) fn peeks(self) -> bool {
        matches!(self, PeekRule::Peek(_))
    }

    /// Whether the player may surrender at all. The EV is a flat -0.5 whenever surrender is offered, on
    /// whichever basis the tree is built, so the solver only needs this boolean.
    pub(crate) fn surrender_offered(self) -> bool {
        match self {
            PeekRule::Peek(s) => s != PeekSurrender::None,
            PeekRule::NoPeek { early_surrender } => early_surrender,
        }
    }

    /// A short label for the surrender option, for display.
    pub(crate) fn surrender_label(self) -> &'static str {
        match self {
            PeekRule::Peek(PeekSurrender::None)
            | PeekRule::NoPeek {
                early_surrender: false,
            } => "none",
            PeekRule::Peek(PeekSurrender::Early)
            | PeekRule::NoPeek {
                early_surrender: true,
            } => "early",
            PeekRule::Peek(PeekSurrender::Late) => "late",
        }
    }
}

/// What a player natural (two-card blackjack) pays, as a multiple of the bet. Modelled as an enum
/// rather than an `f64` because only a tiny set of discrete payouts is ever used: this keeps the value
/// `Eq + Hash` so the enclosing [`Ruleset`] can *derive* those (and key a cache) instead of hand-rolling
/// bit-comparison around a float, and lets call sites match on the rule instead of fuzzy `==` on a float.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum BjPayout {
    /// 3:2 — the good, near-universal payout.
    ThreeToTwo,
    /// 6:5 — a strictly worse rule found on some tables.
    SixToFive,
}

impl BjPayout {
    /// The payout as a multiple of the bet, the form [`resolve_ev`](crate::simulation::resolve_ev)
    /// consumes.
    pub(crate) fn multiplier(self) -> f64 {
        match self {
            BjPayout::ThreeToTwo => 1.5,
            BjPayout::SixToFive => 1.2,
        }
    }

    /// A short label for display.
    pub(crate) fn label(self) -> &'static str {
        match self {
            BjPayout::ThreeToTwo => "3:2",
            BjPayout::SixToFive => "6:5",
        }
    }
}

/// How split aces may be played, as a *single* 3-level axis because the two underlying permissions —
/// drawing more than one card to a split ace, and re-splitting a drawn ace — are coupled: re-splitting
/// aces is incoherent when each split ace receives exactly one card (there is no decision node to
/// re-split at), so the only meaningful settings are the three points on an increasing-liberality
/// ladder. Bundling them lets the type system rule out the incoherent "one card *and* re-split"
/// combination, the same way [`PeekRule`] rules out no-peek late surrender.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum SplitAces {
    /// Each split ace draws exactly one card and stands — no hitting, no doubling, no re-splitting.
    /// The near-universal rule, and the default.
    OneCard,
    /// Split aces may draw and play out like any other split arm (hit, and double when DAS allows),
    /// but a drawn ace may *not* be re-split.
    NoResplit,
    /// Split aces play out fully and a drawn ace may itself be re-split (subject to the hand cap).
    Resplit,
}

impl SplitAces {
    /// Whether a split ace may take more than its forced first card — i.e. play out (hit, and double
    /// when DAS allows) rather than standing on one card. False only for [`SplitAces::OneCard`].
    pub(crate) fn draws_more(self) -> bool {
        self != SplitAces::OneCard
    }

    /// Whether a split ace that draws another ace may be re-split. Only [`SplitAces::Resplit`] allows it
    /// (and it implies [`draws_more`](Self::draws_more), since a re-split arm goes on to draw).
    pub(crate) fn resplit(self) -> bool {
        self == SplitAces::Resplit
    }

    /// A short label for display.
    pub(crate) fn label(self) -> &'static str {
        match self {
            SplitAces::OneCard => "one card",
            SplitAces::NoResplit => "no resplit",
            SplitAces::Resplit => "resplit",
        }
    }
}

/// The stipulation of miscellaneous rules other than the number of decks (?).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct Ruleset {
    /// Whether the dealer hits soft 17
    pub(crate) hs17: bool,
    /// Allowed to double after split
    pub(crate) das: bool,
    /// The dealer's hole-card peek and the player's surrender option (see [`PeekRule`]; they share an
    /// axis because late surrender requires a peek). Off peek, a dealer natural revealed at the end
    /// takes doubled and split bets too, not just the original.
    pub(crate) peek: PeekRule,
    /// What a player natural (a two-card blackjack) pays (see [`BjPayout`]). Only ever applies to a
    /// genuine first-deal natural — a split-arm 21 is *not* a blackjack and pays even money regardless
    /// (see [`arm_stand_ev`](crate::split::arm_stand_ev)).
    pub(crate) bj_payout: BjPayout,
    /// Maximum number of hands the player may end up with after splitting (so the number of splits
    /// allowed is `max_split_hands - 1`). Caps the split recursion — and is what keeps the infinite
    /// deck terminating, since otherwise a pair could be re-split without bound. Setting it to
    /// `4 * n_decks` recovers unbounded splitting on a finite shoe. `< 2` disables splitting.
    pub(crate) max_split_hands: u8,
    /// Split accuracy as a single budget on the **total number of cards drawn** (across all arms)
    /// that are tracked with exact cross-arm depletion. While the budget lasts the depleting shoe
    /// carries forward between arms (each arm sees the cards earlier arms removed — the true
    /// finite-shoe correlation); once this many cards have been drawn, further arms restart from the
    /// pristine post-split shoe (within-arm depletion stays exact throughout — it is cheap and
    /// self-contained; only the expensive cross-arm linkage is truncated).
    ///
    /// One total-cards cap because it sets the truncation *order* directly: a draw path of n cards has
    /// probability ~1/13ⁿ, so capping at K neglects the cross-arm correction uniformly at ~1/13^(K+1),
    /// and the budget is spent first on the shallow, high-probability draws where the correction is
    /// largest. The carried-shoe diversity is bounded by ≤K-card removals from the post-split shoe, a
    /// count independent of deck size, so a small K stays tractable even on 8 decks where a full search
    /// is infeasible. `0` is the old independent-arms approximation (no cross-arm correlation); a K
    /// larger than any reachable draw count (e.g. `u8::MAX`) never resets and so is the full exact
    /// search — combinatorially infeasible on a big shoe, so it is only used for single-query test
    /// validation, never exposed as a chart option. The default `4` is ~5–10× more accurate than
    /// independent (sub-1e-4 vs the exact value) while staying sub-second per query.
    pub(crate) split_cards: u8,
    /// How split aces may be played (see [`SplitAces`]): forced one card (default), hit-but-no-resplit,
    /// or full re-split. Aces are special-cased at most tables because their soft-21 potential is so
    /// strong; this is the axis that relaxes the restriction. ("No double on split aces / tens" would be
    /// a further refinement of `das` per split rank — still future work.)
    pub(crate) split_aces: SplitAces,
}

impl Default for Ruleset {
    fn default() -> Self {
        Self {
            hs17: true,
            das: true,
            peek: PeekRule::Peek(PeekSurrender::Late),
            bj_payout: BjPayout::ThreeToTwo,
            max_split_hands: 4,
            // This is technically not a ruleset option, but a computational precision vs.
            // investment option that specifies the depth of the exact enumeration in multiple
            // splits.
            split_cards: 4,
            split_aces: SplitAces::OneCard,
        }
    }
}
