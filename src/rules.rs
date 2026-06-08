//! Rule configuration: the [`Ruleset`] knobs (and the [`SurrenderRule`] axis) that parametrise the
//! solver. Pure data — no compute lives here. CLAUDE.md anticipates these becoming first-class and
//! threaded through the dealer/payoff logic in place of hardcoded defaults.

/// When (if ever) the player may forfeit half the bet instead of playing the hand out.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum SurrenderRule {
    /// Surrender is not offered.
    None,
    /// Surrender *before* the dealer peeks for blackjack, escaping the dealer-natural loss too.
    /// EV is an unconditional -0.5.
    Early,
    /// Surrender *after* the dealer peeks and shows no blackjack. Only coherent when the dealer
    /// actually peeks (`dealer_check`), since otherwise there is no "after the check".
    Late,
}

/// The stipulation of miscellaneous rules other than the number of decks (?).
pub(crate) struct Ruleset {
    /// Whether the dealer hits soft 17
    pub(crate) hs17: bool,
    /// Allowed to double after split
    pub(crate) das: bool,
    /// Whether the dealer checks their hole card for blackjack
    /// Note that the worst version of this being false causes a dealer blackjack to take
    /// all splits and doubles.
    pub(crate) dealer_check: bool,
    // /// Double on anything (as opposed to just 10 and 11) -- maybe just assume true
    // doa: bool,
    /// What a player natural (a two-card blackjack) pays, as a multiple of the bet. The good and
    /// near-universal value is `1.5` (3:2); some tables pay `1.2` (6:5), a strictly worse rule.
    /// Only ever applies to a genuine first-deal natural — a split-arm 21 is *not* a blackjack and
    /// pays even money regardless (see [`arm_stand_ev`](crate::split::arm_stand_ev)).
    pub(crate) bj_payout: f64,
    /// Whether (and when) the player may surrender.
    pub(crate) surrender: SurrenderRule,
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
    /// larger than any reachable draw count (see [`Ruleset::EXACT_SPLIT`]) never resets and so is the
    /// full exact search. The default `4` is ~5–10× more accurate than independent (sub-1e-4 vs the
    /// exact value) while staying sub-second per query.
    pub(crate) split_cards: u8,
    // TODO: finer split-aces rules. Currently split aces always get exactly one card and cannot be
    // re-split (the common rule); a future axis could relax either, and "no double on split aces /
    // tens" would refine `das` per split rank — see the `SplitSolver` field comments.
}

impl Ruleset {
    /// A `split_cards` budget larger than any draw count a split can reach, so the cross-arm
    /// truncation never fires: the full exact split search (every drawn card tracked, all arms). Not a
    /// magic sentinel — it is just a large `K` that never decrements to `0` in practice.
    /// Combinatorially infeasible on a big shoe — use it for single-query validation, not whole-chart
    /// builds. See [`Ruleset::split_cards`].
    #[allow(dead_code)] // public knob; currently only the tests construct an exact ruleset
    pub(crate) const EXACT_SPLIT: u8 = u8::MAX;

    /// Reject rule combinations that don't correspond to a real game. Late surrender is defined as
    /// surrendering after the dealer peeks, so it only makes sense when the dealer peeks at all.
    pub(crate) fn validate(&self) {
        if self.surrender == SurrenderRule::Late {
            assert!(
                self.dealer_check,
                "Late surrender requires the dealer to peek for blackjack (dealer_check); \
                 use SurrenderRule::Early for a no-peek game."
            );
        }
    }
}

impl Default for Ruleset {
    fn default() -> Self {
        Self {
            hs17: true,
            das: true,
            dealer_check: true,
            bj_payout: 1.5,
            surrender: SurrenderRule::Late,
            max_split_hands: 4,
            // This is technically not a ruleset option, but a computational precision vs.
            // investment option that specifies the depth of the exact enumeration in multiple
            // splits.
            split_cards: 4,
        }
    }
}
