//! The counting **systems** themselves: the [`CountSystem`] compile-time trait, its concrete impls
//! ([`Ko`]/[`HiLo`]/[`AceFive`]), and the runtime [`CountSystemId`] seam (+ [`dispatch_system!`]) that
//! recovers a system's type from a runtime enum. Adding a counting system is local to this file: add a
//! struct + `impl CountSystem`, a [`CountSystemId`] variant, a [`dispatch_system!`] arm, and an entry in
//! [`CountSystemId::ALL`]. The conditions/frames the engine conditions on live in the parent
//! [`count`](crate::count) module.

use serde::{Deserialize, Serialize};

use crate::{card::Card, shoe::CardCol};

/// Which *family* a counting system belongs to — the one robust distinguisher the count-conditioning
/// engine branches on, rather than sniffing the card→value map. A [`Running`](CountKind::Running)
/// system (KO) is actioned on the raw integer running count; a [`TrueCount`](CountKind::TrueCount)
/// system (Hi-Lo) is actioned on the *true count* = running count ÷ decks remaining, so its constraint
/// is a joint inequality in `(running count, remaining-pool size)` rather than a threshold on the
/// running count alone (see [`CountCondition`](crate::count::CountCondition)). Carried in the
/// count-dependent cache keys so two systems can never alias the same persisted solve.
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
    /// The minimal balanced Ace-Five running count ([`AceFive`]).
    AceFive,
}

/// Run `$body` monomorphized over the concrete [`CountSystem`] a runtime [`CountSystemId`] names. A
/// runtime enum value cannot *be* a compile-time type, so a generic call (`solve_counted::<S>(..)`)
/// has to recover the type through a variant→type `match` somewhere — this macro is the single place
/// that `match` lives. `dispatch_system!(id, S => expr)` binds the type `S` to [`Ko`]/[`HiLo`]/
/// [`AceFive`] in turn and evaluates `expr` in each arm. Adding a counting system means adding one arm
/// here (and a [`CountSystemId::ALL`] entry); the call sites stay system-agnostic.
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
            $crate::count::CountSystemId::AceFive => {
                type $S = $crate::count::AceFive;
                $body
            }
        }
    };
}
pub(crate) use dispatch_system;

impl CountSystemId {
    /// Every system in canonical menu order — the single source of truth for "which systems exist" and
    /// the order the count modal cycles through. Adding a system means appending it here (plus its
    /// [`dispatch_system!`] arm); the menu wiring is pure index math over this array and needs no edit.
    pub(crate) const ALL: [CountSystemId; 3] = [
        CountSystemId::Ko,
        CountSystemId::HiLo,
        CountSystemId::AceFive,
    ];

    /// The system `delta` steps along [`ALL`](Self::ALL) from this one, wrapping around: `+1` is the
    /// next system, `−1` the previous. Lets the menu handle both directions as index arithmetic rather
    /// than a hand-written per-direction match.
    pub(crate) fn cycle(self, delta: i32) -> CountSystemId {
        let cur = Self::ALL
            .iter()
            .position(|&s| s == self)
            .expect("every CountSystemId is listed in ALL");
        let len = Self::ALL.len() as i32;
        Self::ALL[(cur as i32 + delta).rem_euclid(len) as usize]
    }

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

    /// The total count value of a full `n`-deck shoe ([`CountSystem::full_shoe_count`]); `0` iff the
    /// system is balanced. Surfaced for the F1 count-description panel.
    pub(crate) fn full_shoe_count(self, n_decks: u8) -> i16 {
        dispatch_system!(self, S => S::full_shoe_count(n_decks))
    }

    /// The system's pivot constant ([`CountSystem::pivot`]) — `external = pivot − internal`. `+4` for
    /// KO, `0` for any balanced system. Surfaced for the F1 count-description panel.
    pub(crate) fn pivot(self, n_decks: u8) -> i16 {
        dispatch_system!(self, S => S::pivot(n_decks))
    }

    /// Whether the system is balanced (every `+v` rank matched by a `−v` rank, so a full shoe counts to
    /// `0`). Balanced systems have pivot `0` and need a true-count division to act on; KO does not.
    pub(crate) fn balanced(self, n_decks: u8) -> bool {
        self.full_shoe_count(n_decks) == 0
    }

    /// Short display name: `KO`, `Hi-Lo`, or `Ace-Five`.
    pub(crate) fn label(self) -> &'static str {
        match self {
            CountSystemId::Ko => "KO",
            CountSystemId::HiLo => "Hi-Lo",
            CountSystemId::AceFive => "Ace-Five",
        }
    }

    /// Free-text usage notes and caveats for the F1 count-description panel, one entry per rendered
    /// line. Kept here beside the system definitions so a new system's caveats land in one place. The
    /// Ace-Five 8-deck warning mirrors the quantitative finding recorded in `CLAUDE.md` and the
    /// `ace_five_edge_by_penetration` measurement in `src/tui/index.rs`.
    pub(crate) fn notes(self) -> &'static [&'static str] {
        match self {
            CountSystemId::Ko => &[
                "Unbalanced: the IRC offsets by deck count so the pivot is a fixed +4.",
                "Act on the running count directly — no true-count division.",
            ],
            CountSystemId::HiLo => &[
                "Balanced: divide the running count by decks remaining for the true count, and act
on that. Indices are quoted in true count.",
            ],
            CountSystemId::AceFive => &[
                "Minimal training-wheels count - only Aces (-1) and 5s (+1).",
                "Balanced, but played as a raw running count (no true-count division).",
                "6-deck: edge is ~break-even at the key count even early",
                "8-deck warning: a fixed RC overrates the edge early in the shoe. At the key count
the first ~third of the shoe is still ~-0.14% EV. Be conservative early — only raise your bet past
about half the shoe.",
            ],
        }
    }
}

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
    fn map(card: &Card) -> i16;

    /// Total count value of a full `n`-deck shoe, `F = Σ_r v_r · f_r`. Zero for balanced systems
    /// (every `+v` rank is matched by a `−v` rank); `+4n` for KO.
    fn full_shoe_count(n_decks: u8) -> i16 {
        CardCol::from_decks(n_decks)
            .iter()
            .map(|(card, quant)| Self::map(&card) * quant as i16)
            .sum()
    }

    /// The system's pivot constant `P = starting_count(n) + full_shoe_count(n)`. This is the
    /// one number the internal⇄external conversion turns on: `external = P − internal`. (KO: `4`;
    /// any balanced system: `0`.)
    fn pivot(n_decks: u8) -> i16 {
        Self::starting_count(n_decks) + Self::full_shoe_count(n_decks)
    }

    /// Convert the player's *external* running count to the deck's *internal* count. This is the
    /// bridge the solver needs: the DP conditions on the internal count, while the player only ever
    /// knows the external one. [`CountShoe::from_external`](crate::countshoe::CountShoe) (the TUI's
    /// entry point) routes through here, so it is the one production home of the conversion. Inverse of
    /// [`internal_to_external`].
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

/// The extremely simple Ace-Five count. This is balanced, but not a true count.
///
/// **8-deck footgun (training-wheels caveat).** Because it is read as a *raw running count* (no
/// true-count division), a fixed running count reads a *higher* true count the deeper the shoe is
/// dealt, so the edge at a fixed RC climbs with penetration and the penetration-marginal key count
/// overstates the early-shoe edge. On 6 decks this is harmless (the edge at the key count is
/// ~break-even even at the start of the shoe); on **8 decks it backfires** — at the key count the
/// first ~third of the shoe is still ~−0.14% EV. See the quantitative table in `CLAUDE.md` and the
/// `ace_five_edge_by_penetration` measurement in [`crate::tui`]. Mitigation that keeps it
/// division-free: only ramp the bet past ~half the shoe (surfaced in the F1 count-description notes).
pub(crate) struct AceFive {}

impl CountSystem for AceFive {
    const KIND: CountKind = CountKind::Running;

    fn map(card: &Card) -> i16 {
        match card {
            Card::Ace => -1,
            Card::Pip(5) => 1,
            _ => 0,
        }
    }
}
