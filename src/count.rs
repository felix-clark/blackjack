//! Definitions and systems of counting
//!
//! NOTE: I think that we might be able to do each "count" independently if we focus on the
//! "pre-deal" count, i.e. the count before the player's initial hand and the dealer's card are
//! shown. The realistic count would include the up-cards as well, so building a count-dependent
//! strategy table from this would need to look across multiple "pre-deal" EV charts to yield the
//! results for a given post-deal count. It's complicated by the fact that, to get precise results,
//! we need to track both the few exactly-known up-cards that impact the total count, as well as a
//! total count that marginalizes over all other possibilities with that constraint.

use serde::{Deserialize, Serialize};

use crate::{card::Card, shoe::CardCol};

/// Which *family* a counting system belongs to — the one robust distinguisher the count-conditioning
/// engine branches on, rather than sniffing the card→value map. A [`Running`](CountKind::Running)
/// system (KO) is actioned on the raw integer running count; a [`TrueCount`](CountKind::TrueCount)
/// system (Hi-Lo) is actioned on the *true count* = running count ÷ decks remaining, so its constraint
/// is a joint inequality in `(running count, remaining-pool size)` rather than a threshold on the
/// running count alone (see [`CountCondition`]). Carried in the count-dependent cache keys so two
/// systems can never alias the same persisted solve.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub(crate) enum CountKind {
    /// Actioned on the raw running count (KO). The condition is a threshold on the internal count `s`.
    Running,
    /// Actioned on the true count = running ÷ decks remaining (Hi-Lo). Necessarily *balanced*
    /// (`full_shoe_count == 0`); the condition is a joint inequality in `(s, n)`.
    TrueCount,
}

/// A concrete counting **system** selected at runtime, as opposed to its [`CountKind`] *family*. The
/// generic [`CountSystem`] trait is the compile-time home of each system's behavior — its card→value
/// [`map`](CountSystem::map), its IRC [`starting_count`](CountSystem::starting_count), and its
/// [`KIND`](CountSystem::KIND); this enum is the runtime seam for the call sites (the trainer) that pick
/// a system at runtime instead of monomorphizing on it. Every method here delegates to the underlying
/// [`CountSystem`] impl, so a system *determines* its family ([`kind`](Self::kind)) and never the
/// reverse — the inversion the family→system [`CountKind::representative_system`] default deliberately
/// isolates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub(crate) enum CountSystemId {
    /// The unbalanced knock-out running count ([`Ko`]).
    Ko,
    /// The balanced Hi-Lo true count ([`HiLo`]).
    HiLo,
}

/// Run `$body` monomorphized over the concrete [`CountSystem`] a runtime [`CountSystemId`] names. A
/// runtime enum value cannot *be* a compile-time type, so a generic call (`solve_counted::<S>(..)`)
/// has to recover the type through a variant→type `match` somewhere — this macro is the single place
/// that `match` lives. `dispatch_system!(id, S => expr)` binds the type `S` to [`Ko`]/[`HiLo`] in turn
/// and evaluates `expr` in each arm. Adding a counting system means adding one arm here and nowhere
/// else; the call sites stay system-agnostic.
macro_rules! dispatch_system {
    ($id:expr, $S:ident => $body:expr) => {
        match $id {
            $crate::count::CountSystemId::Ko => {
                type $S = $crate::count::Ko;
                $body
            }
            $crate::count::CountSystemId::HiLo => {
                type $S = $crate::count::HiLo;
                $body
            }
        }
    };
}
pub(crate) use dispatch_system;

impl CountSystemId {
    /// The family this system is actioned under ([`CountSystem::KIND`]) — running vs. true count. Read
    /// straight off the system, the direction the type system already enforces at compile time.
    pub(crate) fn kind(self) -> CountKind {
        dispatch_system!(self, S => S::KIND)
    }

    /// The count value of `card` under this system ([`CountSystem::map`]).
    pub(crate) fn map(self, card: &Card) -> i16 {
        dispatch_system!(self, S => S::map(card))
    }

    /// The initial running count (IRC) a fresh `n`-deck shoe starts at under this system
    /// ([`CountSystem::starting_count`]).
    pub(crate) fn starting_count(self, n_decks: u8) -> i16 {
        dispatch_system!(self, S => S::starting_count(n_decks))
    }

    /// Short display name: `KO` or `Hi-Lo`.
    pub(crate) fn label(self) -> &'static str {
        match self {
            CountSystemId::Ko => "KO",
            CountSystemId::HiLo => "Hi-Lo",
        }
    }
}

// NOTE: If this ends up not needing to be anything more than a mapping, we can ditch the trait
// formalism and just pass in an arbitrary function Card -> i8 to CountState.
pub(crate) trait CountSystem {
    /// Which count family this system belongs to. [`Running`](CountKind::Running) systems (KO) are
    /// actioned on the raw running count; [`TrueCount`](CountKind::TrueCount) systems (Hi-Lo) on the
    /// true count. The engine branches on this rather than inspecting [`map`](CountSystem::map).
    const KIND: CountKind;

    /// The initial running count the *player* starts from (the system's IRC). Zero for balanced
    /// counts, so the default is implemented. Unbalanced systems (e.g. KO) offset by deck count.
    fn starting_count(_n_decks: u8) -> i16 {
        0
    }

    /// The mapping from card to count.
    /// NOTE: i16 is used because the space is necessary for total counts, and it's easier to
    /// maintain the per-card counts as well.
    /// NOTE: We could also implement this as an array of i8s, of length 10, corresponding to the
    /// internal CardCol array. This would probably optimize.
    fn map(card: &Card) -> i16;

    /// Total count value of a full `n`-deck shoe, `F = Σ_r v_r · f_r`. Zero for balanced systems
    /// (every `+v` rank is matched by a `−v` rank); `+4n` for KO.
    fn full_shoe_count(n_decks: u8) -> i16 {
        CardCol::from_decks(n_decks)
            .iter()
            .map(|(card, quant)| Self::map(&card) * quant as i16)
            .sum()
    }

    /// The system's **pivot constant** `P = starting_count(n) + full_shoe_count(n)`. This is the
    /// one number the internal⇄external conversion turns on: `external = P − internal`. (KO: `4`;
    /// any balanced system: `0`.)
    fn pivot(n_decks: u8) -> i16 {
        Self::starting_count(n_decks) + Self::full_shoe_count(n_decks)
    }

    /// Convert the player's *external* running count to the deck's *internal* count. This is the
    /// bridge the solver needs: the DP conditions on the internal count, while the player only ever
    /// knows the external one. [`CountShoe::from_external`] (the TUI's entry point) routes through
    /// here, so it is the one production home of the conversion. Inverse of [`internal_to_external`].
    ///
    /// [`internal_to_external`]: CountSystem::internal_to_external
    fn external_to_internal(n_decks: u8, external: i16) -> i16 {
        Self::pivot(n_decks) - external
    }

    /// Convert the deck's *internal* running count (count value of the cards still in the shoe) to
    /// the *external* running count the player tallies. Inverse of [`external_to_internal`]; only the
    /// conversion round-trip cross-checks need this direction, hence `#[cfg(test)]`.
    ///
    /// [`external_to_internal`]: CountSystem::external_to_internal
    #[cfg(test)]
    fn internal_to_external(n_decks: u8, internal: i16) -> i16 {
        Self::pivot(n_decks) - internal
    }
}

/// Cards per deck — the true-count normalizer (`TC = cards_per_deck · external / n`).
pub(crate) const CARDS_PER_DECK: i16 = 52;

/// The fixed-point denominator a true-count `cutoff` is expressed in: `cutoff` is in units of
/// `1/TC_HALF_UNITS` true counts, so the resolution is `1/TC_HALF_UNITS` (a half with the default `2`).
/// Bumping it finer keeps everything integer; the index sweep steps in whole true counts
/// (`TC_HALF_UNITS` of these units). The cross-multiplied predicate factor is `CARDS_PER_DECK ·
/// TC_HALF_UNITS` (see [`CountFrame::accepts`]).
pub(crate) const TC_HALF_UNITS: i16 = 2;

/// A **pure constraint** on a `(running count, pool size)` pair — the player's count condition with no
/// bookkeeping of its own. The evaluation pair, and the decision-point anchoring of the entered count,
/// are the [`CountFrame`]'s job; a `CountCondition` is just the literal inequality, kept `Copy`/`Eq`/
/// `Hash` for use as a cache key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CountCondition {
    /// The internal running count equals exactly this value. (Running-count systems only.)
    Eq(i16),
    /// The internal running count is at least this value. (Running-count systems only.)
    Ge(i16),
    /// The internal running count is at most this value. (Running-count systems only.)
    Le(i16),
    /// **True-count** constraint `TC ≥ cutoff/2`. True-count systems are necessarily **balanced**
    /// (`full_shoe_count == 0`, hence pivot 0), so the true count of a pool with internal running count
    /// `s` and size `n` is `−52·s/n`, and `TC ≥ cutoff/2` cross-multiplies (division-free) to
    /// `−104·s ≥ cutoff·n`. `cutoff` is in **half-TC units** (denominator 2) so a fractional threshold
    /// like `TC ≥ 1.5` is `cutoff = 3`, keeping the predicate integer (a literal `f64` could not be a
    /// cache key).
    TrueGe { cutoff: i16 },
    /// **True-count** constraint `TC ≤ cutoff/2`; the `≤` mirror of [`TrueGe`](CountCondition::TrueGe).
    TrueLe { cutoff: i16 },
}

/// A [`CountCondition`] together with the **decision-point anchor** it is evaluated at: the round's
/// visible cards (`vis_sum` = their count value, `vis_cards` = how many) that the player's entered count
/// already includes, under the Wizard-of-Odds convention. This is the per-shoe frame the solver carries
/// — the condition stays a pure inequality; the visible-card adjustment lives here and is applied to the
/// `(s, n)` pair *before* the condition is tested, never folded into the condition itself.
///
/// For a running count the visible shift is already baked into the entered value (`external − map(U) −
/// k`), so `vis = (0, 0)`; only a true count, whose `TC` depends on the pool size, needs to drop the
/// reconstructed root pool to the decision point before testing. Built by [`cond_for_frame`] (per-frame,
/// WoO) or [`CountFrame::pre_round`] (`vis = (0, 0)`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CountFrame {
    pub(crate) cond: CountCondition,
    pub(crate) vis_sum: i16,
    pub(crate) vis_cards: u16,
}

impl CountFrame {
    /// A frame with no visible offset: the constraint is tested at the reconstructed pool directly. This
    /// is every running-count frame (the shift is in the entered value) and any pre-round true count.
    pub(crate) fn pre_round(cond: CountCondition) -> Self {
        Self {
            cond,
            vis_sum: 0,
            vis_cards: 0,
        }
    }

    /// Whether the reconstructed **root** pair `(root_s, root_n)` = (internal running count, pool size)
    /// satisfies this frame. The visible cards are dropped first — the decision-point unseen pool is
    /// `(root_s − vis_sum, root_n − vis_cards)` — then the pure [`CountCondition`] is tested on it.
    /// Running-count conditions test the count alone (ignoring the size, so the visible-card *count* is
    /// immaterial — a running-count frame carries no visible offset anyway). True-count conditions test
    /// the joint integer inequality (computed in `i32`: the `−104·s` and `cutoff·n` terms can exceed
    /// `i16` on a big shoe); a non-positive decision-point size admits no mass.
    pub(crate) fn accepts(&self, root_s: i32, root_n: i32) -> bool {
        let s = root_s - self.vis_sum as i32;
        let n = root_n - self.vis_cards as i32;
        match self.cond {
            CountCondition::Eq(c) => s == c as i32,
            CountCondition::Ge(c) => s >= c as i32,
            CountCondition::Le(c) => s <= c as i32,
            // −(CARDS_PER_DECK·TC_HALF_UNITS)·s ⋛ cutoff·n is `TC ⋛ cutoff/TC_HALF_UNITS` cross-multiplied.
            CountCondition::TrueGe { cutoff } => {
                let f = (CARDS_PER_DECK * TC_HALF_UNITS) as i32;
                n > 0 && -f * s >= cutoff as i32 * n
            }
            CountCondition::TrueLe { cutoff } => {
                let f = (CARDS_PER_DECK * TC_HALF_UNITS) as i32;
                n > 0 && -f * s <= cutoff as i32 * n
            }
        }
    }
}

/// The prior over how many cards remain in the pool (shoe penetration). This is the one inherent
/// modelling choice in count conditioning: the count pins the *value* of the unseen cards but not how
/// many there are. Kept integer-only so the enclosing shoe stays `Eq`/`Hash`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Penetration {
    /// Flat prior over every remaining pool size that is at least `pct`% of the full pool (i.e. up to
    /// `100 − pct`% penetration). The historical default is `25` (deal no deeper than 75%).
    FlatPastPercent(u8),
    /// The pool size is known exactly: only `k` cards remain. Use when the player tracks penetration.
    /// Currently exercised only by the full-deck equivalence cross-check (it pins the pool size so the
    /// count solve degenerates to the exact finite deck), hence `#[cfg(test)]`; promote it to the
    /// production surface if/when the TUI exposes a known-penetration input.
    #[cfg(test)]
    CardsRemaining(u16),
}

impl Penetration {
    /// Prior weight `p(n)` for a remaining pool size of `n`, given the full pool holds `big_n` cards.
    pub(crate) fn weight(&self, n: u16, big_n: u16) -> f64 {
        match self {
            Penetration::FlatPastPercent(pct) => {
                let cutoff = (big_n as u32 * *pct as u32 / 100) as u16;
                if n >= cutoff { 1.0 } else { 0.0 }
            }
            #[cfg(test)]
            Penetration::CardsRemaining(k) => {
                if n == *k {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The prior after one card is drawn from the pool. A known card count decrements; a fractional
    /// cutoff is recomputed against the (now smaller) pool each time it is consulted, so it is fixed.
    pub(crate) fn after_draw(self) -> Self {
        match self {
            #[cfg(test)]
            Penetration::CardsRemaining(k) => Penetration::CardsRemaining(k.saturating_sub(1)),
            other => other,
        }
    }
}

/// How the player's running count is compared against the entered value (the TUI's count condition).
/// Expressed in the player's *external* running count; the inequality inverts when converted to the
/// deck's internal count (`external ≥ C ⟺ internal ≤ pivot − C`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub(crate) enum CountCmp {
    /// Running count exactly equals the entered value.
    Eq,
    /// Running count is at least the entered value.
    Ge,
    /// Running count is at most the entered value.
    Le,
}

/// The unbalanced knock-out system
pub(crate) struct Ko {}

impl CountSystem for Ko {
    const KIND: CountKind = CountKind::Running;

    /// The initial running count for this system
    fn starting_count(n_decks: u8) -> i16 {
        4 - 4 * n_decks as i16
    }

    fn map(card: &Card) -> i16 {
        match card {
            Card::Ace | Card::Ten => -1,
            Card::Pip(r) => {
                if r <= &7 {
                    1
                } else {
                    0
                }
            }
        }
    }
}

/// The balanced Hi-Lo system
pub(crate) struct HiLo {}

impl CountSystem for HiLo {
    const KIND: CountKind = CountKind::TrueCount;

    fn map(card: &Card) -> i16 {
        match card {
            Card::Ace | Card::Ten => -1,
            Card::Pip(r) => {
                if r <= &6 {
                    1
                } else {
                    0
                }
            }
        }
    }
}

/// The unseen pool's [`CountCondition`] for a player external running count `external` compared with
/// `cmp`, under system `S` and `n_decks`. The conversion inverts inequalities: a player count `≥ C`
/// means the unseen pool's internal count is `≤ pivot − C` (more high cards seen tilts the deck the
/// other way). Shared by [`CountShoe::from_external`] and [`CountShoe::band`].
pub(crate) fn cond_from_external<S: CountSystem>(
    n_decks: u8,
    external: i16,
    cmp: CountCmp,
) -> CountCondition {
    let internal = S::external_to_internal(n_decks, external);
    match cmp {
        CountCmp::Eq => CountCondition::Eq(internal),
        CountCmp::Ge => CountCondition::Le(internal),
        CountCmp::Le => CountCondition::Ge(internal),
    }
}

/// The unseen pool's [`CountCondition`] for the player's entered count `value` compared with `cmp`,
/// dispatched on the system's [`CountKind`]. This is the system-agnostic entry: a
/// [`Running`](CountKind::Running) system reads `value` as the external **running** count (delegating
/// to [`cond_from_external`]); a [`TrueCount`](CountKind::TrueCount) system reads it as the external
/// **true** count and builds the joint `(s, n)` inequality directly (no inversion — the [`TrueGe`]/
/// [`TrueLe`] predicates are already phrased in the player's external TC; true-count systems are
/// balanced, so pivot 0 is baked in).
///
/// True counts are **inequality-only** ([`CountCmp::Eq`] is rejected): an exact true count is a
/// measure-zero event over the `(s, n)` lattice, so only `≥`/`≤` are meaningful. `value` is in
/// **half-TC units** (so `TC ≥ 1.5` is `value = 3`), keeping the whole predicate division-free and the
/// shoe `Eq`/`Hash`. This is the **pre-round** condition; the per-frame Wizard-of-Odds [`CountFrame`]
/// that anchors the TC at the decision point is [`cond_for_frame`].
///
/// [`TrueGe`]: CountCondition::TrueGe
/// [`TrueLe`]: CountCondition::TrueLe
///
/// Test-only: production builds frames through [`cond_for_frame`] (which adds the decision-point
/// offset); this pre-round-only entry survives as a cross-check convenience.
#[cfg(test)]
pub(crate) fn cond_from_count<S: CountSystem>(
    n_decks: u8,
    value: i16,
    cmp: CountCmp,
) -> CountCondition {
    match S::KIND {
        CountKind::Running => cond_from_external::<S>(n_decks, value, cmp),
        CountKind::TrueCount => {
            assert_balanced::<S>(n_decks);
            true_cond(cmp, value)
        }
    }
}

/// The per-frame [`CountFrame`] under the **Wizard-of-Odds** convention that the entered count includes
/// this round's visible cards (the up-card `up` plus a player hand of count value `k`). It is what the
/// 5-frame chart/index merge ([`merge_count_frames`](crate::tui::merge_count_frames)) builds, reading
/// each hand from the frame matching its own count value.
///
/// - [`Running`](CountKind::Running): the player's external running count `value` minus the round's
///   visible count `map(up) + k`, then the existing internal inversion ([`cond_from_external`]). This is
///   exactly the prior KO behavior (`external − map(U) − k`), with no visible offset on the frame.
/// - [`TrueCount`](CountKind::TrueCount): a pure [`TrueGe`](CountCondition::TrueGe)/
///   [`TrueLe`](CountCondition::TrueLe) on the half-unit `value`, paired with `vis_sum = map(up) + k`
///   and the caller's `vis_cards`, so [`CountFrame::accepts`] drops the root pool to the decision point
///   before testing `TC ⋛ value/2`.
///
/// `vis_cards` is supplied by the caller (3 for an up-card + 2-card hand; 1 for the insurance decision,
/// which sees only the up-card).
pub(crate) fn cond_for_frame<S: CountSystem>(
    n_decks: u8,
    value: i16,
    cmp: CountCmp,
    up: Card,
    k: i16,
    vis_cards: u16,
) -> CountFrame {
    let vis_sum = S::map(&up) + k;
    match S::KIND {
        CountKind::Running => {
            CountFrame::pre_round(cond_from_external::<S>(n_decks, value - vis_sum, cmp))
        }
        CountKind::TrueCount => {
            assert_balanced::<S>(n_decks);
            CountFrame {
                cond: true_cond(cmp, value),
                vis_sum,
                vis_cards,
            }
        }
    }
}

/// Build a true-count [`CountCondition`] from a half-unit `cutoff`, rejecting [`CountCmp::Eq`] (true
/// counts are inequality-only). Shared by [`cond_from_count`] and [`cond_for_frame`].
fn true_cond(cmp: CountCmp, cutoff: i16) -> CountCondition {
    match cmp {
        CountCmp::Ge => CountCondition::TrueGe { cutoff },
        CountCmp::Le => CountCondition::TrueLe { cutoff },
        CountCmp::Eq => panic!(
            "true-count systems are inequality-only; CountCmp::Eq is not supported (use Ge/Le)"
        ),
    }
}

/// Assert (debug-only) that `S` is a balanced system. The `True*` predicate hardcodes pivot 0 — the
/// player's true count is `−52·s/n` only when the full shoe count is 0, which every standard true-count
/// system (Hi-Lo, etc.) satisfies.
fn assert_balanced<S: CountSystem>(n_decks: u8) {
    debug_assert_eq!(
        S::pivot(n_decks),
        0,
        "TrueCount systems must be balanced (pivot 0); the True* predicate assumes it"
    );
}
