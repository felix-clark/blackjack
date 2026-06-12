//! Definitions and systems of counting
//!
//! NOTE: I think that we might be able to do each "count" independently if we focus on the
//! "pre-deal" count, i.e. the count before the player's initial hand and the dealer's card are
//! shown. The realistic count would include the up-cards as well, so building a count-dependent
//! strategy table from this would need to look across multiple "pre-deal" EV charts to yield the
//! results for a given post-deal count. It's complicated by the fact that, to get precise results,
//! we need to track both the few exactly-known up-cards that impact the total count, as well as a
//! total count that marginalizes over all other possibilities with that constraint.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::{
    card::Card,
    shoe::{CardCol, N_RANKS, Shoe},
};

/// Shared memo from a pool's **count-class composition** (and its count condition) to the per-rank
/// *value-scale* broadcast — the class-level half of the count-conditioned draw distribution (see
/// [`CountW::value_scales`]). The expensive deconvolution + `(s,n)`-sum depends on the pool *only*
/// through its class composition `(M_class)`, never the within-class rank breakdown; the breakdown
/// re-enters only as the cheap `m_r` multiplier when [`CountW::dist_from_scales`] reconstructs the
/// distribution. So keying on class composition collapses every pool that is class-identical but
/// rank-distinct (e.g. a dealer line that drew a 5 vs. a 6 — both KO +1) onto one deconvolution, on
/// top of the exact-pool dedup it replaces. Every [`CountShoe`] cloned within one solve shares the
/// same `Arc`. The condition is itself a function of the class composition given a fixed starting count
/// (every removed card shifts the threshold by its value), but it is kept in the key for robustness
/// across bands/solves.
///
/// The class composition is encoded as `[u16; N_RANKS]` with each rank holding its *whole class's*
/// pool count (`Σ_{r': v_{r'}=v_r} M_{r'}`, see [`CountW::class_counts`]) — a value identical across
/// class-equivalent pools and distinct otherwise, so it is a faithful class-composition key.
type DistCache = Arc<Mutex<HashMap<([u16; N_RANKS], CountCondition), [f64; N_RANKS]>>>;

// NOTE: If this ends up not needing to be anything more than a mapping, we can ditch the trait
// formalism and just pass in an arbitrary function Card -> i8 to CountState.
pub(crate) trait CountSystem {
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

/// A condition the player imposes on the **internal** running count of the unseen pool (the count of
/// the cards still in the shoe). The solver conditions every draw distribution on this. As cards are
/// drawn the target shifts by the drawn card's value — see [`CountCondition::shifted`] — so the same
/// condition threaded down the tree stays the player's "all visible cards counted" constraint.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum CountCondition {
    /// The internal running count equals exactly this value.
    Eq(i16),
    /// The internal running count is at least this value.
    Ge(i16),
    /// The internal running count is at most this value.
    Le(i16),
}

impl CountCondition {
    /// Whether an internal running count `s` satisfies the condition.
    pub(crate) fn accepts(&self, s: i16) -> bool {
        match self {
            CountCondition::Eq(c) => s == *c,
            CountCondition::Ge(c) => s >= *c,
            CountCondition::Le(c) => s <= *c,
        }
    }

    /// The condition on the *remaining* pool after a card of count value `v` is drawn. Removing a
    /// `+v` card from a pool of count `s` leaves count `s − v`, so every threshold shifts by `−v`.
    pub(crate) fn shifted(&self, v: i16) -> CountCondition {
        match self {
            CountCondition::Eq(c) => CountCondition::Eq(c - v),
            CountCondition::Ge(c) => CountCondition::Ge(c - v),
            CountCondition::Le(c) => CountCondition::Le(c - v),
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
    fn weight(&self, n: u16, big_n: u16) -> f64 {
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
    fn after_draw(self) -> Self {
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

/// A [`Shoe`] whose draws are conditioned on a [`CountCondition`] over the unseen pool's internal
/// running count. This is the value that flows through the solver's hot loop; the expensive DP lives
/// in the wrapped `CountW` table. Drawing a card removes it from the pool *and*
/// shifts the count condition by the card's value (so the same physical constraint — "all visible
/// cards counted = C" — is maintained as the tree descends) and ages the penetration prior.
///
/// `Clone` (not `Copy`) because it owns the [`CountW`] table; the solver's generic bounds are relaxed
/// from `Copy` to `Clone` to admit it. `Eq`/`Hash` (a memo-key requirement) are hand-written over the
/// *defining* fields — pool, value map, condition, penetration — since the `CountW` table and cached
/// `dist` are pure functions of those (and `f64`/`Vec` aren't `Eq`/`Hash` anyway).
#[derive(Clone)]
pub(crate) struct CountShoe {
    /// The current unseen pool (full shoe minus every card drawn so far), maintained cheaply on every
    /// `draw` by a single-rank decrement. This — not [`dp`](Self::dp)'s deck — is the shoe's *true*
    /// pool: it is the source of `m_r` and the class-composition cache key, and what `Eq`/`Hash` and the
    /// `Shoe` queries (`contains_hand`/`rank_count`/`all_draw_probs`) read. The expensive `CountW` table
    /// is only synced down to it lazily, on a draw-distribution cache miss (see [`sync_table`]).
    ///
    /// [`sync_table`]: Self::sync_table
    pool: CardCol,
    /// The count-weight table. Its own `deck` is the pool the table was last deconvolved to — an
    /// *ancestor* of [`pool`](Self::pool) under lazy syncing (they coincide right after a sync or at
    /// construction). Cloning is cheap (`Arc`-shared `w`); the table is materialized to the current pool
    /// only when [`sync_table`](Self::sync_table) runs on a cache miss.
    dp: CountW,
    cond: CountCondition,
    pen: Penetration,
    /// The count-conditioned next-card distribution, cached so `draw_prob`/`all_draw_probs` are O(1);
    /// recomputed once per `draw` (which is when the pool/condition change).
    dist: [f64; N_RANKS],
    /// When set (via [`CountShoe::mean_field_view`], used by the split solver) the shoe behaves as a
    /// plain **finite** deck whose composition `dp.deck` is the *count-tilted expected remaining pool*:
    /// draws deplete it and read hypergeometric probabilities straight off it, with no count
    /// reconditioning. This captures both the count tilt (baked into the composition) and finite
    /// depletion (the deck shrinks) at finite-deck speed — the split-solver order-limit. The `CountW`
    /// table is unused in this mode.
    mean_field: bool,
    /// Units per real card in mean-field mode (`0` otherwise). The mean-field deck is built at high
    /// resolution — total `n · mf_scale` units — so the sub-card count tilt survives rounding; drawing
    /// a card removes `mf_scale` units, keeping depletion exactly `1/n`. See [`expected_composition`].
    mf_scale: u16,
    /// When this shoe is one member of a **count band** (a sweep of external running counts solved
    /// together for the chart's count-index thresholds), the conditions of *all* band members at the
    /// current pool, with `cond == band_conds[band_idx]`. Empty for an ordinary single-count solve.
    /// On a draw-distribution cache miss the whole band is filled from one deconvolution (see
    /// [`CountW::draw_dist_band`]), so sibling members sharing this shoe's `dist_cache` pay the
    /// expensive work once. Shifts in lockstep with `cond` as cards are drawn; excluded from
    /// `Eq`/`Hash` (it is a cache-warming hint, not part of the value the shoe represents — `cond`
    /// already pins that).
    band_conds: Vec<CountCondition>,
    /// This member's index into `band_conds` (0 when not banded).
    band_idx: usize,
    /// Shared `(pool, condition) → draw distribution` memo (see [`DistCache`]). Excluded from
    /// `Eq`/`Hash`/`PartialEq` — it is a pure cache, identical-keyed shoes share it, and it never
    /// affects the value a shoe represents.
    dist_cache: DistCache,
}

impl PartialEq for CountShoe {
    fn eq(&self, other: &Self) -> bool {
        // The current `pool` (not the lazily-lagging table deck) is the defining composition. In
        // mean-field mode the working composition lives in `dp.deck`, so compare that instead.
        let (a, b) = if self.mean_field {
            (&self.dp.deck, &other.dp.deck)
        } else {
            (&self.pool, &other.pool)
        };
        a == b
            && self.dp.value_of_rank == other.dp.value_of_rank
            && self.cond == other.cond
            && self.pen == other.pen
            && self.mean_field == other.mean_field
            && self.mf_scale == other.mf_scale
    }
}
impl Eq for CountShoe {}
impl std::hash::Hash for CountShoe {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        let deck = if self.mean_field {
            &self.dp.deck
        } else {
            &self.pool
        };
        deck.hash(h);
        self.dp.value_of_rank.hash(h);
        self.cond.hash(h);
        self.pen.hash(h);
        self.mean_field.hash(h);
        self.mf_scale.hash(h);
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

impl CountShoe {
    /// A shoe of `n_decks` under counting system `S`, conditioned on `cond` over the *internal*
    /// running count, with penetration prior `pen`. The caller converts a player's external running
    /// count to the internal condition via [`CountSystem::external_to_internal`].
    pub(crate) fn new<S: CountSystem>(n_decks: u8, cond: CountCondition, pen: Penetration) -> Self {
        let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
        let dp = CountW::build(value_of_rank, CardCol::from_decks(n_decks));
        let dist = dp.draw_dist(cond, pen);
        Self::from_parts(dp, cond, pen, dist)
    }

    /// Build from the player's *external* running count and a comparison, converting to the deck's
    /// internal count condition. The conversion inverts inequalities: a player count `≥ C` means the
    /// unseen pool's internal count is `≤ pivot − C` (more cards seen pushes the deck the other way).
    pub(crate) fn from_external<S: CountSystem>(
        n_decks: u8,
        external: i16,
        cmp: CountCmp,
        pen: Penetration,
    ) -> Self {
        Self::new::<S>(
            n_decks,
            cond_from_external::<S>(n_decks, external, cmp),
            pen,
        )
    }

    /// A **band** of shoes, one per external running count in `externals`, that share a single
    /// [`CountW`] build *and* one draw-distribution cache. The shared cache plus each member carrying
    /// the whole `band_conds` list means the expensive per-pool deconvolution is computed once for the
    /// entire band (every member's distribution extracted from one deconvolution — see
    /// [`CountW::draw_dist_band`]) rather than once per member. Solve the first member to completion to
    /// warm the cache; the remaining members are then nearly free (pure cache hits). Used by the TUI to
    /// sweep the count axis for a column's count-index thresholds.
    pub(crate) fn band<S: CountSystem>(
        n_decks: u8,
        externals: &[i16],
        cmp: CountCmp,
        pen: Penetration,
    ) -> Vec<Self> {
        let conds: Vec<CountCondition> = externals
            .iter()
            .map(|&e| cond_from_external::<S>(n_decks, e, cmp))
            .collect();
        let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
        let dp = CountW::build(value_of_rank, CardCol::from_decks(n_decks));
        let cache: DistCache = Arc::new(Mutex::new(HashMap::new()));
        conds
            .iter()
            .enumerate()
            .map(|(band_idx, &cond)| {
                let mut shoe = Self {
                    pool: dp.deck,
                    dp: dp.clone(),
                    cond,
                    pen,
                    dist: [0.0; N_RANKS],
                    mean_field: false,
                    mf_scale: 0,
                    band_conds: conds.clone(),
                    band_idx,
                    dist_cache: cache.clone(),
                };
                shoe.recompute();
                shoe
            })
            .collect()
    }

    fn from_parts(
        dp: CountW,
        cond: CountCondition,
        pen: Penetration,
        dist: [f64; N_RANKS],
    ) -> Self {
        Self {
            pool: dp.deck,
            dp,
            cond,
            pen,
            dist,
            mean_field: false,
            mf_scale: 0,
            band_conds: Vec::new(),
            band_idx: 0,
            dist_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The split solver's [`Shoe::for_split`] target: a finite deck whose composition is the
    /// count-tilted *expected remaining pool*, built at high resolution (`scale` units per real card)
    /// so the sub-card tilt survives. Draws deplete it `scale` units at a time (depletion exactly
    /// `1/n`) and read probabilities straight off it, so the split sub-solve gets both the count tilt
    /// and finite depletion at finite-deck speed (the `CountW` table is left unused). `dp.deck` is
    /// repurposed as that scaled mean-field composition.
    fn mean_field_view(&self) -> Self {
        // The current pool size — `pool`, not the lazily-lagging table deck — is what `self.dist` was
        // conditioned on and what the tilted composition must represent.
        let n = self.pool.len() as u16;
        let (comp, scale) = expected_composition(&self.dist, n);
        let mut next = self.clone();
        // In mean-field mode the `CountW` weight table is never read — draws come straight off the
        // tilted `dp.deck` composition (see `draw`/`draw_prob`/`all_draw_probs`/`remove_hand`). Drop it
        // so the many mean-field shoes the split solver caches (one per reachable composition, in
        // `dealer_cache`/`draw_cache`/`memo`) don't each retain a full `s_span × n_span` table. At 4+
        // decks that retention blew memory into the GBs and got the process SIGKILLed; the table is
        // ~300 KB there and thousands of distinct compositions are reached per split, times the caches
        // and the concurrent up-card columns.
        next.dp.w = Arc::new(Vec::new());
        next.dp.deck = comp;
        next.mean_field = true;
        next.mf_scale = scale;
        next
    }

    /// A `CardCol` holding `mf_scale` copies of each card in `hand` — the scaled multiset removed from
    /// the mean-field deck when `hand` is drawn.
    fn scaled(&self, hand: &CardCol) -> CardCol {
        let mut out = CardCol::new();
        for (card, cnt) in hand.iter() {
            out.add_n(card, cnt * self.mf_scale);
        }
        out
    }

    /// The underlying (un-tilted) finite pool — the full shoe minus the cards removed so far. Running
    /// the split solver on this is exact finite depletion that ignores the count tilt; used by the
    /// `freeze_split_error` measurement to A/B the split approximations, hence `#[cfg(test)]`.
    #[cfg(test)]
    pub(crate) fn pool(&self) -> CardCol {
        self.pool
    }

    /// Deplete the pool by one card *without* recomputing the cached distribution — used to batch a
    /// whole-hand removal before a single recompute. Only the *current pool* (and the count condition /
    /// penetration prior) are updated here — both `O(1)`; the expensive `CountW` table is left
    /// untouched and synced lazily later (see [`sync_table`]). The condition shifts by the card's value
    /// (so the running-count constraint follows the card down the tree) and the penetration prior ages.
    ///
    /// [`sync_table`]: Self::sync_table
    fn deplete(&mut self, card: &Card) {
        let v = self.dp.value_of_rank[card.rank_index()];
        // One card off the current pool (per-rank saturating subtraction). NOT `remove_rank`, which
        // clears the whole rank.
        self.pool = self.pool - CardCol::from_hand(&[*card]);
        self.cond = self.cond.shifted(v);
        // Band members shift in lockstep, so `cond == band_conds[band_idx]` is preserved as the
        // running-count constraint follows the card down the tree.
        for c in self.band_conds.iter_mut() {
            *c = c.shifted(v);
        }
        self.pen = self.pen.after_draw();
    }

    /// Bring the lazily-deferred `CountW` table down to the current [`pool`](Self::pool). `deplete`
    /// advances the pool cheaply on each draw but leaves the table at an ancestor pool; this
    /// deconvolves the difference (the cards drawn since the last sync) so the table matches the current
    /// pool. Called only on a draw-distribution cache *miss*, where the table is actually needed — so a
    /// shoe (and the clones sharing its `Arc`-table) that only ever hits the class-composition cache
    /// never pays the deconvolution at all. The first `remove_card` here copies the shared table
    /// on-write (`Arc::make_mut`); the rest mutate it in place.
    fn sync_table(&mut self) {
        let removed = self.dp.deck - self.pool;
        for (card, n) in removed.iter() {
            for _ in 0..n {
                self.dp.remove_card(&card);
            }
        }
        debug_assert_eq!(self.dp.deck, self.pool, "table not synced to current pool");
    }

    /// Rebuild the cached count-conditioned draw distribution for the current pool/condition. The
    /// shared memo (see [`DistCache`]) caches the class-level [`CountW::value_scales`] keyed on the
    /// *current pool*'s class composition, so the expensive deconvolution runs at most once per distinct
    /// class composition across the whole solve; the cheap rank-level `m_r` fold-in
    /// ([`CountW::dist_from_scales_with`]) is redone per pool. On a cache **miss** the lazily-deferred
    /// table is first [`sync_table`](Self::sync_table)'d down to the current pool; on a **hit** the table
    /// is never touched.
    fn recompute(&mut self) {
        let class_counts = self.dp.class_counts_of(&self.pool);
        let key = (class_counts, self.cond);
        if let Some(&scales) = self.dist_cache.lock().unwrap().get(&key) {
            self.dist = self.dp.dist_from_scales_with(&self.pool, &scales);
            return;
        }
        // Miss: the deconvolution genuinely needs the table at the current pool.
        self.sync_table();
        if self.band_conds.len() <= 1 {
            let scales = self.dp.value_scales(self.cond, self.pen);
            self.dist_cache.lock().unwrap().insert(key, scales);
            self.dist = self.dp.dist_from_scales_with(&self.pool, &scales);
            return;
        }
        // Banded: one deconvolution serves the whole band. Fill every member's value scales for this
        // class composition so sibling solves sharing `dist_cache` hit them instead of re-deconvolving
        // (the band is filled atomically — all present or none — so the early cache check above is a
        // sound guard).
        let scales = self.dp.value_scales_band(&self.band_conds, self.pen);
        {
            let mut cache = self.dist_cache.lock().unwrap();
            for (c, sc) in self.band_conds.iter().zip(scales.iter()) {
                cache.insert((class_counts, *c), *sc);
            }
        }
        self.dist = self
            .dp
            .dist_from_scales_with(&self.pool, &scales[self.band_idx]);
    }
}

/// Build the mean-field "expected remaining composition" deck for a draw distribution `dist` (which
/// sums to 1) representing `n` real cards, returning `(deck, scale)`. The deck is built at high
/// resolution — total `n · scale` units, `counts[r] ≈ dist[r] · n · scale` by the **largest-remainder**
/// method — so the sub-card count tilt is not lost to rounding. The caller draws `scale` units per
/// real card (depletion exactly `1/n`). `scale` is chosen to keep the total comfortably under `u16`.
fn expected_composition(dist: &[f64; N_RANKS], n: u16) -> (CardCol, u16) {
    let n = n.max(1);
    // Total ≈ 50_000 units (well under u16::MAX), so resolution is ~1/50_000 of the deck.
    let scale = (50_000 / n).max(1);
    let total = n * scale;
    let target: [f64; N_RANKS] = std::array::from_fn(|r| dist[r] * total as f64);
    let mut counts = [0u16; N_RANKS];
    for r in 0..N_RANKS {
        counts[r] = target[r].floor() as u16;
    }
    let mut remainder = total.saturating_sub(counts.iter().sum());
    // Hand the leftover units to the largest fractional parts first.
    let mut order: [usize; N_RANKS] = std::array::from_fn(|r| r);
    order.sort_by(|&a, &b| {
        let fa = target[a] - target[a].floor();
        let fb = target[b] - target[b].floor();
        fb.partial_cmp(&fa).unwrap()
    });
    for &r in order.iter() {
        if remainder == 0 {
            break;
        }
        counts[r] += 1;
        remainder -= 1;
    }
    let mut col = CardCol::new();
    for (r, &count) in counts.iter().enumerate() {
        col.add_n(Card::from_rank_index(r), count);
    }
    (col, scale)
}

impl Shoe for CountShoe {
    fn draw(&mut self, card: &Card) {
        if self.mean_field {
            // Finite depletion on the tilted composition (one real card = `mf_scale` units); no count
            // reconditioning.
            self.dp.deck = self.dp.deck - self.scaled(&CardCol::from_hand(&[*card]));
            return;
        }
        self.deplete(card);
        self.recompute();
    }

    fn draw_prob(&self, card: &Card) -> f64 {
        if self.mean_field {
            let total = self.dp.deck.len() as f64;
            return if total > 0.0 {
                self.dp.deck.get_count(card) as f64 / total
            } else {
                0.0
            };
        }
        self.dist[card.rank_index()]
    }

    fn all_draw_probs(&self) -> impl Iterator<Item = (Card, f64)> {
        // In both modes: yield only ranks present in the working composition (matching
        // `CardCol::all_draw_probs`) — a depleted rank would otherwise make the player Hit DP look up an
        // un-enumerable child. In mean-field mode that composition (and the weights) is the tilted
        // `dp.deck`; otherwise it is the current `pool`, with the count-tilted `dist` for weights.
        let mean_field = self.mean_field;
        let dist = self.dist;
        let deck = if mean_field { self.dp.deck } else { self.pool };
        let total = deck.len() as f64;
        (0..N_RANKS)
            .filter(move |&i| deck.get_count_i(i) > 0)
            .map(move |i| {
                let p = if mean_field {
                    deck.get_count_i(i) as f64 / total
                } else {
                    dist[i]
                };
                (Card::from_rank_index(i), p)
            })
    }

    fn remove_hand(&self, hand: &CardCol) -> Self {
        if self.mean_field {
            let mut next = self.clone();
            next.dp.deck = next.dp.deck - self.scaled(hand);
            return next;
        }
        let mut next = self.clone();
        for (card, n) in hand.iter() {
            for _ in 0..n {
                next.deplete(&card);
            }
        }
        next.recompute();
        next
    }

    fn contains_hand(&self, hand: &CardCol) -> bool {
        if self.mean_field {
            return self.scaled(hand).is_submultiset(&self.dp.deck);
        }
        hand.is_submultiset(&self.pool)
    }

    /// The enumerator's per-rank supply is the pool count. The hypergeometric scan-weight is not the
    /// count-conditioned joint and is treated as a loose pooling weight (see `simulation.rs`); the
    /// count-correct probabilities come from [`draw_prob`](Self::draw_prob)/[`all_draw_probs`]. The
    /// supply is the current `pool` (the mean-field split view reads its tilted `dp.deck` instead).
    fn rank_count(&self, rank: &Card) -> Option<u16> {
        let deck = if self.mean_field {
            &self.dp.deck
        } else {
            &self.pool
        };
        Some(deck.get_count(rank))
    }

    fn for_split(&self) -> Self {
        self.mean_field_view()
    }
}

/// Incremental count table, held in **normalized probability space**:
///
/// ```text
///   p[s][n] = P( a uniformly random n-subset of the current pool has running count s )
///           = ( Σ_{count-s size-n subsets} 1 ) / C(N, n)
/// ```
///
/// where `N` is the current pool size. Every entry is a probability in `[0, 1]`, so the table is
/// numerically stable at any deck size — unlike the raw subset *counts* `W[s][n]` it used to store,
/// which at ≥2 decks blow past `f64`'s exact-integer range (`C(208,104) ≈ 1e61`) and, combined with
/// the `1/C(N,n) ≈ 1e-61` normalizer applied separately, destroyed the draw distribution through
/// catastrophic cancellation (a negative/`inf` total then escaping the `> 0` guard unnormalized).
///
/// Flattened over (running count `s`, remaining size `n`). Supports **O(cells) single-card removal**
/// by a normalized deconvolution (the probability table of the pool minus one card), so the dealer and
/// player recursions build it once per shoe and deplete it incrementally instead of rebuilding per
/// draw. Each fold/deconv step is a convex (or near-convex, bounded by `N/(N−n)`) recombination of
/// adjacent cells, so values never leave `O(1)`. The recurrences are derived from the raw
/// generating-function identities (`W'[s][n] = W[s][n] + W[s−v][n−1]` for adding a card; its inverse
/// for removing one) by carrying through the per-`n` `C(N,n)` normalizer — so they are algebraically
/// identical to the old raw DP and reproduce the 1-deck oracle bit-for-bit, only without the overflow.
#[derive(Clone)]
struct CountW {
    /// per-rank count value `v_r`, indexed like [`CardCol`].
    value_of_rank: [i16; N_RANKS],
    /// the current pool (per-rank `M_r`); a class size `M_j` is the sum of `M_r` over ranks sharing
    /// value `v_j`.
    deck: CardCol,
    s_min: i16,
    s_max: i16,
    /// the current pool size `N` — the largest `n` that carries weight; shrinks by 1 per card removed.
    n_max: u16,
    n_span: usize,
    /// flat `[s][n]` table of normalized probabilities; `p[s][n]` lives at `(s − s_min) * n_span + n`.
    ///
    /// `Arc`-wrapped so cloning a [`CountShoe`] (every solver clone / `remove_hand`) is a refcount bump
    /// rather than a ~300 KB deep copy. The table is only *mutated* lazily — when a draw-distribution
    /// cache miss forces [`CountShoe::sync_table`] to deconvolve it down to the current pool — and the
    /// mutation copies-on-write via `Arc::make_mut`, so the deep copy happens at most once per shoe that
    /// actually has to deconvolve, never per clone.
    w: Arc<Vec<f64>>,
}

impl CountW {
    fn at(&self, s: i16, n: u16) -> f64 {
        if s < self.s_min || s > self.s_max {
            return 0.0;
        }
        self.w[(s - self.s_min) as usize * self.n_span + n as usize]
    }

    /// Build the table for `deck` under the per-rank value map `value_of_rank`, folding the pool one
    /// card at a time (each fold grows the sub-pool by one card, in normalized space).
    fn build(value_of_rank: [i16; N_RANKS], deck: CardCol) -> Self {
        let s_min: i16 = (0..N_RANKS)
            .map(|r| (value_of_rank[r] * deck.get_count_i(r) as i16).min(0))
            .sum();
        let s_max: i16 = (0..N_RANKS)
            .map(|r| (value_of_rank[r] * deck.get_count_i(r) as i16).max(0))
            .sum();
        let n_max = deck.len() as u16;
        let n_span = n_max as usize + 1;
        let s_span = (s_max - s_min + 1) as usize;
        let mut w = vec![0.0; s_span * n_span];
        // Seed the empty sub-pool: the only 0-subset has count 0, with probability 1.
        w[(0 - s_min) as usize * n_span] = 1.0;

        let mut me = Self {
            value_of_rank,
            deck,
            s_min,
            s_max,
            n_max,
            n_span,
            w: Arc::new(w),
        };
        // Fold cards in one at a time, tracking the growing sub-pool size.
        let mut size: u16 = 0;
        for (r, &v) in value_of_rank.iter().enumerate() {
            for _ in 0..deck.get_count_i(r) {
                size += 1;
                me.fold_in_card(v, size);
            }
        }
        me
    }

    /// Add one card of value `v` to a sub-pool of size `n_new − 1`, bringing it to size `n_new`, in
    /// normalized space. The raw recurrence `W'[s][n] = W[s][n] + W[s−v][n−1]` becomes, after dividing
    /// each side by its `C(·, n)` normalizer, the **convex** combination
    ///
    /// ```text
    ///   p'[s][n] = p[s][n] · (N − n)/N  +  p[s−v][n−1] · n/N        (N = n_new)
    /// ```
    ///
    /// both coefficients in `[0,1]` and summing to 1, so probabilities stay bounded. Iterating `n`
    /// downward keeps the `p[s−v][n−1]` term at its pre-update (size `N−1`) value.
    fn fold_in_card(&mut self, v: i16, n_new: u16) {
        let big_n = n_new as f64;
        for n in (1..=n_new).rev() {
            let keep = (n_new - n) as f64 / big_n; // (N − n)/N
            let take = n as f64 / big_n; //            n/N
            for s in self.s_min..=self.s_max {
                let here = self.at(s, n);
                let prev = self.at(s - v, n - 1);
                let val = here * keep + prev * take;
                if val != 0.0 || here != 0.0 {
                    // Unique during `build` (the only caller), so `make_mut` never actually clones.
                    Arc::make_mut(&mut self.w)
                        [(s - self.s_min) as usize * self.n_span + n as usize] = val;
                }
            }
        }
    }

    /// The normalized probability table for the **current pool minus one card of value `v`** (size
    /// `M = N − 1`). The inverse of [`fold_in_card`]: from `p` (size `N`) recover `p'` via
    ///
    /// ```text
    ///   p'[s][n] = p[s][n] · N/(N − n)  −  p'[s−v][n−1] · n/(N − n)
    /// ```
    ///
    /// sweeping `n` upward so the `n−1` term is already final. The amplifier `N/(N − n)` is mild for
    /// small `n` (`≤ 2` up to the half-size) but grows toward `N` near the top, where the subtraction
    /// loses precision and the row-sums drift (≈1% at the very top of a 1-deck table). So we run the
    /// recurrence only over the stable lower half `n ∈ 0..=M/2` and recover the upper half by
    /// **complement reflection**: an `n`-subset of the reduced pool and its `(M − n)`-subset complement
    /// partition it, so their counts sum to the pool's total count `S'`, giving
    /// `p'[s][n] = p'[S' − s][M − n]` exactly. Every reflected index lands in the already-computed
    /// stable region, so the whole table stays accurate at any deck size.
    fn deconv(&self, v: i16) -> Vec<f64> {
        let big_n = self.n_max as f64;
        let m = self.n_max - 1; // size of the reduced pool (N − 1); 0 when N == 1
        let half = m / 2;
        // Total count of the reduced pool: the full pool's running count, minus the removed card.
        let pool_count: i16 = (0..N_RANKS)
            .map(|r| self.value_of_rank[r] * self.deck.get_count_i(r) as i16)
            .sum();
        let s_prime = pool_count - v;

        let mut h = vec![0.0; self.w.len()];
        // Stable lower half via the recurrence (amplifier ≤ 2 here).
        for n in 0..=half {
            let denom = (self.n_max - n) as f64; // N − n  (> 0 since n ≤ M/2 < N)
            let keep = big_n / denom; //              N/(N − n)
            let take = n as f64 / denom; //           n/(N − n)
            for s in self.s_min..=self.s_max {
                let idx = (s - self.s_min) as usize * self.n_span + n as usize;
                let sub = if n == 0 || s - v < self.s_min || s - v > self.s_max {
                    0.0
                } else {
                    h[(s - v - self.s_min) as usize * self.n_span + (n - 1) as usize]
                };
                h[idx] = self.w[idx] * keep - sub * take;
            }
        }
        // Upper half by complement reflection: p'[s][n] = p'[S' − s][M − n].
        for n in (half + 1)..=m {
            for s in self.s_min..=self.s_max {
                let mirror_s = s_prime - s;
                let mirror = if mirror_s < self.s_min || mirror_s > self.s_max {
                    0.0
                } else {
                    h[(mirror_s - self.s_min) as usize * self.n_span + (m - n) as usize]
                };
                h[(s - self.s_min) as usize * self.n_span + n as usize] = mirror;
            }
        }
        h
    }

    /// Remove one card of `rank` from the pool, depleting the table *in place* (no allocation — this is
    /// the dealer recursion's hot path). Applies the same normalized deconvolution as [`deconv`] but
    /// overwrites `w` directly: the lower-half recurrence sweeps `n` upward, reading each cell's
    /// still-old full-pool value and the already-rewritten `n−1` row; the upper-half reflection then
    /// reads those finalized lower rows. The stale top size-`N` row is left as-is — it is never read
    /// again once `n_max` shrinks. O(cells), no allocation. Mirrors [`deconv`] exactly otherwise.
    fn remove_card(&mut self, rank: &Card) {
        let v = self.value_of_rank[rank.rank_index()];
        let big_n = self.n_max as f64;
        let m = self.n_max - 1;
        let half = m / 2;
        let pool_count: i16 = (0..N_RANKS)
            .map(|r| self.value_of_rank[r] * self.deck.get_count_i(r) as i16)
            .sum();
        let s_prime = pool_count - v;

        // Copy-on-write the table once for the whole removal: if `w` is `Arc`-shared with sibling
        // shoes, `make_mut` clones it here (the deferred deep copy that lazy syncing pays only on a
        // cache miss); thereafter it is unique and the loops mutate in place. `n_span`/`s_min` are read
        // into locals first so `w` can hold the sole borrow of `self.w`.
        let n_span = self.n_span;
        let s_min = self.s_min;
        let w = Arc::make_mut(&mut self.w);

        // Lower half via the recurrence, in place (sweep n upward).
        for n in 0..=half {
            let denom = (self.n_max - n) as f64;
            let keep = big_n / denom;
            let take = n as f64 / denom;
            for s in s_min..=self.s_max {
                let idx = (s - s_min) as usize * n_span + n as usize;
                let sub = if n == 0 || s - v < s_min || s - v > self.s_max {
                    0.0
                } else {
                    w[(s - v - s_min) as usize * n_span + (n - 1) as usize]
                };
                w[idx] = w[idx] * keep - sub * take;
            }
        }
        // Upper half by complement reflection off the now-finalized lower rows.
        for n in (half + 1)..=m {
            for s in s_min..=self.s_max {
                let mirror_s = s_prime - s;
                let mirror = if mirror_s < s_min || mirror_s > self.s_max {
                    0.0
                } else {
                    w[(mirror_s - s_min) as usize * n_span + (m - n) as usize]
                };
                w[(s - s_min) as usize * n_span + n as usize] = mirror;
            }
        }

        self.deck = self.deck - CardCol::from_hand(&[*rank]);
        self.n_max -= 1;
    }

    /// The pool's **count-class composition**, encoded per rank: `class_counts_of(pool)[r]` is the total
    /// count in `pool` of *every* rank sharing rank `r`'s count value (`Σ_{r': v_{r'}=v_r} M_{r'}`). This
    /// is the [`DistCache`] key — identical across pools that are class-equivalent but rank-distinct
    /// (the very pools whose [`value_scales`](Self::value_scales) agree), and distinct otherwise — so
    /// caching on it collapses the redundant deconvolutions those pools would each otherwise pay.
    ///
    /// The key is read off the shoe's *current* pool — which, under lazy table syncing, runs ahead of
    /// the table's (deferred) deck — so the pool is passed explicitly; only `value_of_rank` (the fixed
    /// count system) comes from the table.
    fn class_counts_of(&self, pool: &CardCol) -> [u16; N_RANKS] {
        std::array::from_fn(|r| {
            let v = self.value_of_rank[r];
            (0..N_RANKS)
                .filter(|&r2| self.value_of_rank[r2] == v)
                .map(|r2| pool.get_count_i(r2))
                .sum()
        })
    }

    /// The **class-level half** of the count-conditioned draw distribution: per rank `r`, the value
    /// scale `scale_{v_r} = Σ_{s: cond} Σ_{n=1..=N} pen(n) · p_v[s − v][n − 1]`, where `p_v` is the
    /// normalized count table of the pool with one `v`-card removed ([`deconv`](Self::deconv)).
    ///
    /// This is the expensive part (one deconvolution + `(s,n)`-sum per distinct present value), and it
    /// depends on the pool **only through its class composition** — `deconv`, `s_min/s_max`, `n_max`,
    /// the table, the `cond` filter and the `pen` weight are all functions of the per-class sizes, never
    /// the within-class rank breakdown. So it is shared across class-equivalent pools (keyed by
    /// [`class_counts_of`](Self::class_counts_of)). The scale is broadcast to *every* rank of a present class,
    /// not just the ranks currently in the pool, so the result is a pure function of the class
    /// composition (a momentarily-absent rank must not change it). [`dist_from_scales`] folds in the
    /// rank-level `m_r` to recover the distribution.
    ///
    /// [`dist_from_scales`]: Self::dist_from_scales
    fn value_scales(&self, cond: CountCondition, pen: Penetration) -> [f64; N_RANKS] {
        let big_n = self.n_max;
        let mut out = [0.0; N_RANKS];
        let mut seen_values: Vec<i16> = Vec::new();
        let mut scale_of_value: Vec<f64> = Vec::new();
        for (r, &v) in self.value_of_rank.iter().enumerate() {
            // Only a *present* class can be deconvolved (it needs ≥1 card of value `v`) and only a
            // present class is ever drawn; an absent class keeps scale 0 (all its ranks have `m_r = 0`,
            // so they contribute nothing once `m_r` is folded in). Broadcasting to every rank of a
            // present class — even ranks momentarily absent from this pool — is what keeps the result a
            // pure function of the class composition, so class-equivalent pools share a cache entry.
            let class_present =
                (0..N_RANKS).any(|r2| self.value_of_rank[r2] == v && self.deck.get_count_i(r2) > 0);
            if !class_present {
                continue;
            }
            let scale = if let Some(pos) = seen_values.iter().position(|&u| u == v) {
                scale_of_value[pos]
            } else {
                let p_v = self.deconv(v);
                let mut s_v = 0.0;
                for s in self.s_min..=self.s_max {
                    if !cond.accepts(s) {
                        continue;
                    }
                    for n in 1..=big_n {
                        let w_n = pen.weight(n, big_n);
                        if w_n == 0.0 {
                            continue;
                        }
                        let sh = s - v;
                        if sh < self.s_min || sh > self.s_max {
                            continue;
                        }
                        let pval = p_v[(sh - self.s_min) as usize * self.n_span + (n - 1) as usize];
                        if pval != 0.0 {
                            s_v += w_n * pval;
                        }
                    }
                }
                seen_values.push(v);
                scale_of_value.push(s_v);
                s_v
            };
            out[r] = scale;
        }
        out
    }

    /// Fold the rank-level multiplicities `m_r` into a value-scale broadcast ([`value_scales`]) and
    /// normalize, recovering the count-conditioned next-card distribution: `acc[r] = m_r · scale_{v_r}`,
    /// renormalized to sum 1. This is the raw `Σ pen(n)/C(N,n)/n · M_r · H_v[s−v][n−1]` of the old DP
    /// with the `1/(C(N,n)·n)` folded into `p_v` (via `C(N−1,n−1)/C(N,n) = n/N`, the common `1/N`
    /// cancelling in the normalization) — same value, computed entirely on `O(1)` probabilities.
    ///
    /// A finite, positive total is the reachable case. If the condition is unreachable for this pool
    /// (no admissible config) the total is 0 and the all-zero distribution is returned as-is — callers
    /// treat a zero draw distribution as "this line of play has no count-consistent mass".
    ///
    /// [`value_scales`]: Self::value_scales
    fn dist_from_scales(&self, scales: &[f64; N_RANKS]) -> [f64; N_RANKS] {
        self.dist_from_scales_with(&self.deck, scales)
    }

    /// [`dist_from_scales`](Self::dist_from_scales) folding in the `m_r` of an arbitrary `pool`. The
    /// reconstruction is read off the shoe's *current* pool (which, under lazy syncing, may run ahead of
    /// the table's deck), so the multiplicities are taken from `pool`; the `scales` are class-level and
    /// independent of which pool within the class composition they came from.
    fn dist_from_scales_with(&self, pool: &CardCol, scales: &[f64; N_RANKS]) -> [f64; N_RANKS] {
        let mut acc: [f64; N_RANKS] =
            std::array::from_fn(|r| pool.get_count_i(r) as f64 * scales[r]);
        let total: f64 = acc.iter().sum();
        if total > 0.0 && total.is_finite() {
            acc.iter_mut().for_each(|p| *p /= total);
        }
        acc
    }

    /// Next-card draw distribution conditioned on `cond` over the running count, under penetration
    /// prior `pen`. The class-level [`value_scales`](Self::value_scales) (cached across class-equivalent
    /// pools) folded together with the rank-level `m_r` by [`dist_from_scales`](Self::dist_from_scales);
    /// kept as a single entry point for the oracle cross-checks.
    fn draw_dist(&self, cond: CountCondition, pen: Penetration) -> [f64; N_RANKS] {
        self.dist_from_scales(&self.value_scales(cond, pen))
    }

    /// [`value_scales`] for a whole **band** of conditions at once, sharing the expensive work.
    ///
    /// The condition only enters through the `cond.accepts(s)` filter on the running count `s`; the
    /// `O(cells)` deconvolution `p_v` and the `n`-sum over the penetration prior are both
    /// condition-*independent*. So for a band of conditions over the *same* pool we deconvolve each
    /// present value once and the per-`s` `n`-sum once, then distribute that mass to whichever band
    /// members admit `s`. Returns one value-scale broadcast per `cond`, in order — equal to calling
    /// [`value_scales`] per condition (same terms, only the `n`-sum regrouped per-`s` so float
    /// summation order differs in the last bits), at a fraction of the cost. This is what lets the
    /// solver sweep a count band (the chart's count-index thresholds) while paying the dominant
    /// deconvolution only once.
    ///
    /// [`value_scales`]: Self::value_scales
    fn value_scales_band(&self, conds: &[CountCondition], pen: Penetration) -> Vec<[f64; N_RANKS]> {
        let big_n = self.n_max;
        let mut outs = vec![[0.0; N_RANKS]; conds.len()];
        let mut seen_values: Vec<i16> = Vec::new();
        // Per distinct present value, the `s_v` scale for each band member (parallel to `conds`).
        let mut scales_of_value: Vec<Vec<f64>> = Vec::new();
        for r in 0..N_RANKS {
            let v = self.value_of_rank[r];
            // Present-class guard + broadcast to every rank of the class — see `value_scales`.
            let class_present =
                (0..N_RANKS).any(|r2| self.value_of_rank[r2] == v && self.deck.get_count_i(r2) > 0);
            if !class_present {
                continue;
            }
            let pos = if let Some(pos) = seen_values.iter().position(|&u| u == v) {
                pos
            } else {
                let p_v = self.deconv(v);
                let mut sv = vec![0.0; conds.len()];
                for s in self.s_min..=self.s_max {
                    let sh = s - v;
                    if sh < self.s_min || sh > self.s_max {
                        continue;
                    }
                    // The `n`-sum is condition-independent; compute it once per `s` and hand it to
                    // every member whose condition admits `s`.
                    let mut nsum = 0.0;
                    for n in 1..=big_n {
                        let w_n = pen.weight(n, big_n);
                        if w_n == 0.0 {
                            continue;
                        }
                        let pval = p_v[(sh - self.s_min) as usize * self.n_span + (n - 1) as usize];
                        if pval != 0.0 {
                            nsum += w_n * pval;
                        }
                    }
                    if nsum != 0.0 {
                        for (ci, cond) in conds.iter().enumerate() {
                            if cond.accepts(s) {
                                sv[ci] += nsum;
                            }
                        }
                    }
                }
                seen_values.push(v);
                scales_of_value.push(sv);
                scales_of_value.len() - 1
            };
            for (ci, out) in outs.iter_mut().enumerate() {
                out[r] = scales_of_value[pos][ci];
            }
        }
        outs
    }

    /// [`draw_dist`] for a whole band of conditions at once — [`value_scales_band`] with the rank-level
    /// `m_r` folded in and each member normalized. Returns one distribution per `cond`, in order.
    /// Production reconstructs distributions straight from [`value_scales_band`] (in `recompute`); this
    /// is retained only as the band/per-cond cross-check oracle, hence `#[cfg(test)]`.
    ///
    /// [`draw_dist`]: Self::draw_dist
    /// [`value_scales_band`]: Self::value_scales_band
    #[cfg(test)]
    fn draw_dist_band(&self, conds: &[CountCondition], pen: Penetration) -> Vec<[f64; N_RANKS]> {
        self.value_scales_band(conds, pen)
            .iter()
            .map(|sc| self.dist_from_scales(sc))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;
    use std::collections::HashMap;

    /// Binomial coefficient C(n, k) as `f64`, multiplicative form. Only the test-only enumeration
    /// oracles ([`CountState`]/[`CountDp`]) need it now — the production [`CountW`] runs entirely in
    /// normalized probability space and never forms a raw binomial. Evaluated as `∏ (n−k+i)/i`,
    /// multiplying before dividing so each partial product is an exact integer binomial; the smaller
    /// leg is chosen by symmetry. Exact up to `2^53`, which the 1-deck oracle stays under.
    fn choose(n: u16, k: u16) -> f64 {
        if k > n {
            return 0.;
        }
        let k = k.min(n - k);
        (1..=k).fold(1.0, |acc, i| acc * (n - k + i) as f64 / i as f64)
    }

    /// Wall-clock for a single count-conditioned up-card column (run with
    /// `--release --ignored --nocapture`). The worst case is a low up-card like a 6, whose deep dealer
    /// draw tree is where the count-conditioned draw-distribution work concentrates; the dealer-recursion
    /// reorder ([`crate::dealer`]) and the `(pool, condition)` draw-distribution memo ([`DistCache`])
    /// keep it tractable. A rough perf guard, not a correctness check.
    #[test]
    #[ignore]
    fn bench_count_column() {
        use crate::rules::Ruleset;
        use crate::simulation::build_evs;
        use std::time::Instant;

        let rules = Ruleset::default();
        for n in [2u8, 4u8] {
            for up in [Card::Pip(6), Card::Ten] {
                let shoe = CountShoe::from_external::<Ko>(
                    n,
                    0,
                    CountCmp::Eq,
                    Penetration::FlatPastPercent(25),
                );
                let start = Instant::now();
                let tree = build_evs(shoe, up, &rules);
                eprintln!(
                    "n={n} up={up:?}: {:?}  tree={}",
                    start.elapsed(),
                    tree.len()
                );
            }
        }
    }

    /// The normalized [`CountW`] DP must stay numerically sane at **multiple decks**, where the old
    /// raw-coefficient table overflowed `f64`'s exact-integer range and silently emitted unnormalized
    /// (`~1e60`, even `inf`) draw distributions — the cause of the giant-EV / mostly-Stand / `unwrap`
    /// crash when a count was imposed on a 2+ deck shoe. The 1-deck oracle cross-checks
    /// ([`count_w_draw_dist_matches_oracle`]) stay under `2^53` and so never caught this; this guard
    /// exercises 2 and 4 decks directly.
    ///
    /// At every node the count-conditioned next-card distribution must be a genuine probability
    /// distribution: non-negative, finite, and summing to either 1 (a reachable count) or exactly 0 (an
    /// unreachable count — the honest "no count-consistent mass" signal). It must never be unnormalized.
    #[test]
    fn count_w_multideck_draw_dist_normalized() {
        let val: [i16; N_RANKS] = std::array::from_fn(|r| Ko::map(&Card::from_rank_index(r)));
        let pen = Penetration::FlatPastPercent(25);
        for n in [2u8, 4u8] {
            // Walk a depleting pool down a dealer-style draw chain so `remove_card` (the in-place
            // deconvolution) is exercised, not just a fresh `build`.
            let mut cw = CountW::build(val, CardCol::from_decks(n));
            for step in 0..6 {
                for cond in [
                    CountCondition::Eq(0),
                    CountCondition::Eq(Ko::full_shoe_count(n)), // fresh-shoe count, always reachable
                    CountCondition::Ge(-5),
                    CountCondition::Le(5),
                    CountCondition::Eq(10_000), // wildly out of range → unreachable
                ] {
                    let dist = cw.draw_dist(cond, pen);
                    let sum: f64 = dist.iter().sum();
                    assert!(
                        dist.iter().all(|p| p.is_finite() && *p >= -1e-12),
                        "n={n} step={step} {cond:?}: non-finite or negative draw prob: {dist:?}"
                    );
                    assert!(
                        sum.abs() < 1e-9 || (sum - 1.0).abs() < 1e-9,
                        "n={n} step={step} {cond:?}: draw dist sums to {sum}, not 0 or 1"
                    );
                }
                // Deplete one card (a Ten, the deepest-count rank) and continue.
                if cw.deck.get_count(&Card::Ten) > 0 {
                    cw.remove_card(&Card::Ten);
                }
            }
        }
    }

    /// [`CountW::draw_dist_band`] must reproduce [`CountW::draw_dist`] for every member of the band:
    /// the band path shares one deconvolution across all conditions, so this pins that the sharing is
    /// value-preserving (it must agree to floating-point, the only difference being the regrouped
    /// `n`-sum). Walks a depleting pool so the in-place `remove_card` tables are covered too.
    #[test]
    fn count_w_draw_dist_band_matches_single() {
        let val: [i16; N_RANKS] = std::array::from_fn(|r| Ko::map(&Card::from_rank_index(r)));
        let pen = Penetration::FlatPastPercent(25);
        let conds = [
            CountCondition::Eq(0),
            CountCondition::Eq(3),
            CountCondition::Ge(-2),
            CountCondition::Le(4),
            CountCondition::Eq(9_999), // unreachable → all-zero on both paths
        ];
        for n in [1u8, 2u8] {
            let mut cw = CountW::build(val, CardCol::from_decks(n));
            for step in 0..5 {
                let band = cw.draw_dist_band(&conds, pen);
                for (i, &cond) in conds.iter().enumerate() {
                    let single = cw.draw_dist(cond, pen);
                    for r in 0..N_RANKS {
                        assert!(
                            (band[i][r] - single[r]).abs() < 1e-12,
                            "n={n} step={step} {cond:?} rank {r}: band {} vs single {}",
                            band[i][r],
                            single[r]
                        );
                    }
                }
                if cw.deck.get_count(&Card::Ten) > 0 {
                    cw.remove_card(&Card::Ten);
                }
            }
        }
    }

    /// A [`CountShoe::band`] member must behave exactly like the standalone [`CountShoe::from_external`]
    /// for its own external count: same draw distribution after the same depletion. This is what makes
    /// a band sweep sound — each layer is the very solve the single-count chart would do, only with the
    /// deconvolution shared. (The band member draws its distribution through `draw_dist_band` + the
    /// shared cache; the singleton through `draw_dist`.)
    #[test]
    fn count_shoe_band_matches_singletons() {
        use crate::shoe::Shoe;
        let pen = Penetration::FlatPastPercent(25);
        let externals = [-4i16, -1, 0, 2, 5];
        let hand = CardCol::from_hand(&[Card::Ten, Card::Pip(6), Card::Pip(5)]);
        for n in [1u8, 2u8] {
            let band = CountShoe::band::<Ko>(n, &externals, CountCmp::Eq, pen);
            for (k, &ext) in externals.iter().enumerate() {
                let single = CountShoe::from_external::<Ko>(n, ext, CountCmp::Eq, pen);
                let b = band[k].remove_hand(&hand);
                let s = single.remove_hand(&hand);
                for r in 0..N_RANKS {
                    let card = Card::from_rank_index(r);
                    assert!(
                        (b.draw_prob(&card) - s.draw_prob(&card)).abs() < 1e-12,
                        "n={n} ext={ext} rank {r}: band {} vs single {}",
                        b.draw_prob(&card),
                        s.draw_prob(&card)
                    );
                }
            }
        }
    }

    /// Eager enumeration reference for the count-conditioned draw distribution. Superseded in production
    /// by [`CountW`] (the incremental table threaded through [`CountShoe`]); retained `#[cfg(test)]` as the
    /// oracle the `CountW`/DP cross-checks pin against, and as the home of the standalone conversion tests.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct CountState {
        /// The unknown pool: the full shoe minus any exactly-known removed cards. Every pool size `M_j`
        /// and count probability below is taken over this.
        deck: CardCol,
        /// The counting system materialized as a per-rank value `v_r`, indexed like `CardCol`'s inner
        /// array (`rank_index`). This *is* the whole system; the class grouping, pool sizes, and total
        /// counts are all derived from it together with `deck` via [`count_classes`](Self::count_classes).
        value_of_rank: [i16; N_RANKS],
        /// Number of decks in the *full* shoe this state was configured for. Note this is not
        /// recoverable from `deck`, which may be a depleted pool — it is fixed at construction and is
        /// what the external⇄internal conversion is calibrated to.
        n_decks: u8,
        /// The system's pivot constant `P` for `n_decks` (see [`CountSystem::pivot`]). Captured here so
        /// the conversion does not have to re-derive the IRC, which the bare `value_of_rank` mapping
        /// discards.
        pivot: i16,
    }

    impl CountState {
        /// This state's per-rank count values grouped into classes (the DP-friendly representation).
        fn count_classes(&self) -> CountClasses {
            CountClasses::from_value_map(self.value_of_rank, &self.deck)
        }

        /// Configure a state for a `deck` (which may be a depleted pool) drawn from an `n_decks` shoe.
        /// Generic over the whole `CountSystem`, not just its mapping, so the IRC-derived `pivot` is
        /// captured here rather than silently dropped.
        fn from_pool<S: CountSystem>(deck: CardCol, n_decks: u8) -> Self {
            // The system is deck-independent, so materialize it for every rank (not just ranks present
            // in `deck`): a rank that is momentarily absent still has a well-defined count value.
            let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
            Self {
                deck,
                value_of_rank,
                n_decks,
                pivot: S::pivot(n_decks),
            }
        }

        pub fn from_decks<S: CountSystem>(n: u8) -> Self {
            Self::from_pool::<S>(CardCol::from_decks(n), n)
        }

        /// Convert this state's internal running count to the player's external count, and back. These
        /// are the instance-level conveniences: the shoe size and pivot are already baked in, so the
        /// caller never re-passes `n_decks` (the affine identity itself lives on [`CountSystem`]).
        pub(crate) fn internal_to_external(&self, internal: i16) -> i16 {
            self.pivot - internal
        }

        pub(crate) fn external_to_internal(&self, external: i16) -> i16 {
            self.pivot - external
        }

        /// All draw probabilities given a running count (eventually an arbitrary function over (k_j)).
        ///
        /// This is the eager enumeration reference path. The production path is `draw_probs_where`
        /// (via the `CountDp` dynamic program); `dp_matches_enumeration` pins the two together.
        pub(crate) fn all_draw_probs_given_c(&self, c: i16) -> impl Iterator<Item = (Card, f64)> {
            let classes = self.count_classes();
            let mut card_prob_array = [0.; N_RANKS];
            for (knums, kprob) in self.prob_val_counts_given_c(c) {
                // n: total cards in the shoe for this class-count configuration.
                let n = knums.iter().sum::<u16>() as f64;
                // For a single (next-card) draw, the probability that the card has rank r factors as
                //
                //     P(rank r | config) = P(class j) · P(rank r | class j)
                //                        = (k_j / n)  ·  (M_r / M_j),
                //
                // where rank r lives in count class j:
                //   * k_j / n    is the fraction of the remaining shoe sitting in class j;
                //   * M_r / M_j  is the within-class fraction of rank r — the remaining k_j cards of
                //     class j are a uniform subset of the M_j full-shoe cards of that class, so by the
                //     hypergeometric mean the expected fraction that are rank r is M_r / M_j (exact for
                //     a single draw, by linearity of expectation).
                // These sum to 1 over r within each config: Σ_j (k_j/n)(M_j/M_j) = Σ_j k_j/n = 1.
                for (r, &j) in classes.class_of_rank.iter().enumerate() {
                    // `knums` is indexed by class (same order as `classes.values`/`sizes`), so the class
                    // index is the only mapping needed — no value→index lookup.
                    let m_r = self.deck.get_count_i(r) as f64;
                    let m_j = classes.sizes[j] as f64;
                    let k_j = knums[j] as f64;
                    // Weight P(rank r | config) by the (unnormalized) probability of the config itself.
                    card_prob_array[r] += kprob * (k_j / n) * (m_r / m_j);
                }
            }
            // Each config's draw distribution already sums to 1, so this normalization only divides out
            // the unnormalized config weight Σ kprob (and guards against floating-point drift). It will
            // produce NaNs if `c` is unreachable (empty config map) — see TODO on prob_val_counts_given_c.
            let card_prob_norm: f64 = card_prob_array.iter().sum();
            card_prob_array
                .iter_mut()
                .for_each(|p| *p /= card_prob_norm);
            card_prob_array
                .into_iter()
                .enumerate()
                .map(|(i, p)| (Card::from_rank_index(i), p))
        }

        /// Count-conditioned next-card draw distribution, in full generality.
        ///
        /// `accept(s, n)` keeps the `(internal running count, remaining total)` cells consistent with
        /// what the player knows — equality (`s == c`), an inequality (`s >= c`), a true-count bin (any
        /// function of both `s` and `n`), etc. `n_weight(n)` is the prior `p(n)` over shoe depth; return
        /// `0` below the penetration cutoff. Result is `P(next card = rank)` indexed by rank.
        ///
        /// For each accepted cell the config probability is `∝ p(n)/C(N, n) · ∏_j C(M_j, k_j)`, and a
        /// rank `r` in class `j` is drawn with probability `(k_j / n) · (M_r / M_j)`. Summed over the
        /// configs in a cell, the `∏ C · k_j` part is exactly the stored moment `T_j[s][n]`, so the cell
        /// contributes `p(n)/C(N, n) · (T_j[s][n] / n) · (M_r / M_j)` — no per-config work.
        fn draw_probs_where(
            &self,
            accept: impl Fn(i16, u16) -> bool,
            n_weight: impl Fn(u16) -> f64,
        ) -> [f64; N_RANKS] {
            let classes = self.count_classes();
            let dp = CountDp::build(&classes);
            let big_n = dp.n_max;

            // The normalizer `1/C(N, n)` depends only on `n`; precompute the row so it isn't recomputed
            // for every `s`. (`choose(big_n, 0) = 1`; index 0 is unused since the `n` loop starts at 1.)
            let inv_cn: Vec<f64> = (0..=dp.n_max).map(|n| 1.0 / choose(big_n, n)).collect();

            let mut rank_probs = [0.0; N_RANKS];
            for s in dp.s_min..=dp.s_max {
                for n in 1..=dp.n_max {
                    if !accept(s, n) {
                        continue;
                    }
                    let w_n = n_weight(n);
                    if w_n == 0.0 {
                        continue;
                    }
                    let base = dp.cell(s, n);
                    if dp.data[base] == 0.0 {
                        continue; // unreachable count/total combination
                    }
                    // p(n) · 1/C(N, n); the 1/n of the draw probability is applied per rank below.
                    let cell_w = w_n * inv_cn[n as usize];
                    for (r, rank_prob_r) in rank_probs.iter_mut().enumerate() {
                        let j = classes.class_of_rank[r];
                        let m_j = classes.sizes[j];
                        if m_j == 0 {
                            continue;
                        }
                        let m_r = self.deck.get_count_i(r) as f64;
                        let t_j = dp.data[base + 1 + j]; // T_j[s][n] = Σ_configs k_j · ∏ C
                        // acc[r] += cell_w * (t_j / n as f64) * (m_r / m_j as f64);
                        *rank_prob_r += cell_w * (t_j / n as f64) * (m_r / m_j as f64);
                    }
                }
            }

            // Each cell's contribution already forms a sub-distribution over ranks, so normalizing just
            // divides out the total accepted mass (and guards FP drift). Stays all-zero if the condition
            // is unreachable, rather than producing NaNs.
            let total: f64 = rank_probs.iter().sum();
            if total > 0.0 {
                rank_probs.iter_mut().for_each(|p| *p /= total);
            }
            rank_probs
        }

        /// Draw distribution conditioned on the internal running count equalling `c`, under a flat prior
        /// over shoe depth past 75% penetration (matching the `all_draw_probs_given_c` reference path).
        fn draw_probs_given_count(&self, c: i16) -> [f64; N_RANKS] {
            let cutoff = self.deck.len() as u16 / 4;
            self.draw_probs_where(|s, _| s == c, move |n| if n >= cutoff { 1.0 } else { 0.0 })
        }

        /// The multivariate hypergeometric pmf giving the probability that the deck holds an exact
        /// per-class count vector `(k_0, …, k_{m-1})` (class order = sorted count value) with running
        /// count `c`. These are the numbers of each class in the shoe right now, so may be reversed from
        /// the external count. Conditioned on a deck-size distribution that for now cuts off below 75%
        /// penetration.
        ///
        /// This is the enumeration oracle for `draw_probs_where`; it is intentionally eager. See
        /// [`CountDp`] for the production dynamic program over the same quantity.
        ///
        /// TODO: generalize the fixed `c` to an arbitrary predicate on the count vector (needed for
        /// relative systems like Hi-Lo and for inequalities like `C >= X`); the DP path already takes
        /// such a predicate via `accept`.
        fn prob_val_counts_given_c(&self, c: i16) -> HashMap<Vec<u16>, f64> {
            let classes = self.count_classes();
            let big_n = self.deck.len() as u16;
            // Don't draw into more than 75% of the deck.
            let n_cutoff = big_n / 4;

            classes
                .sizes
                .iter()
                .map(|&m| 0..=m)
                .multi_cartesian_product()
                .filter_map(|knums| {
                    // Total cards in the shoe for this configuration.
                    let n_k = knums.iter().sum::<u16>();
                    if n_k < n_cutoff {
                        return None;
                    }
                    // Running count of this configuration: Σ_j v_j k_j.
                    let count: i16 = classes
                        .values
                        .iter()
                        .zip(&knums)
                        .map(|(&v, &k)| v * k as i16)
                        .sum();
                    // TODO: replace with an arbitrary predicate on `knums` (see method doc).
                    if count != c {
                        return None;
                    }
                    // ∏_j C(M_j, k_j) is the multivariate-hypergeometric numerator; dividing by C(N, n_k)
                    // applies the uniform-n prior normalizer (load-bearing across n, see the n-prior note
                    // in the module docs). A richer p(n) would multiply in here instead of the flat cutoff.
                    let weight = knums
                        .iter()
                        .zip(&classes.sizes)
                        .map(|(&k, &m)| choose(m, k))
                        .product::<f64>()
                        / choose(big_n, n_k);
                    Some((knums, weight))
                })
                .collect()
        }

        /// The total running count of the cards currently in the deck. This is the deck's internal
        /// count, not the one used by the player.
        pub(crate) fn running_count(&self) -> i16 {
            self.deck
                .iter()
                .map(|(card, quant)| self.value_of_rank[card.rank_index()] * quant as i16)
                .sum()
        }
    }

    /// A counting system reduced to its action on a specific deck: the ten ranks grouped into classes
    /// by shared count value. Grouping is exact for the count distribution (the remaining count of a
    /// union of ranks is hypergeometric with the union's pool size), and it is what lets the DP's count
    /// axis depend on `m <= 10` class values rather than on all ten ranks individually.
    struct CountClasses {
        /// class index -> count value `v_j` (integer; fractional systems pre-scale, e.g. Halves x2).
        values: Vec<i16>,
        /// class index -> pool size `M_j` (cards in the deck carrying that value).
        sizes: Vec<u16>,
        /// rank index -> class index. Subsumes the old `count_index`.
        class_of_rank: [usize; N_RANKS],
    }

    impl CountClasses {
        /// Group the ranks of `deck` by their count value. `deck` is the *unknown pool* (the full shoe
        /// minus any known up-cards), so `M_j` and the conditioned count refer to what is actually
        /// uncertain.
        fn from_value_map(values: [i16; N_RANKS], deck: &CardCol) -> Self {
            // Distinct count values, sorted for a stable class order.
            let mut distinct: Vec<i16> = values.to_vec();
            distinct.sort_unstable();
            distinct.dedup();

            let class_of_rank: [usize; N_RANKS] =
                std::array::from_fn(|r| distinct.iter().position(|&v| v == values[r]).unwrap());

            let mut sizes = vec![0u16; distinct.len()];
            for r in 0..N_RANKS {
                sizes[class_of_rank[r]] += deck.get_count_i(r);
            }

            Self {
                values: distinct,
                sizes,
                class_of_rank,
            }
        }
    }

    /// Dynamic-programming table over (running count `s`, remaining card total `n`) for a count system.
    ///
    /// Each cell carries `1 + m` accumulators (`m` = number of count classes), laid out contiguously
    /// with stride `width`:
    ///   slot `0`     : `W[s][n]   = Σ_configs ∏_j C(M_j, k_j)`
    ///   slot `1 + j` : `T_j[s][n] = Σ_configs k_j · ∏_l C(M_l, k_l)`   (first moment of class `j`)
    /// where the sums run over every class-count configuration `(k_0..k_{m-1})` whose running count is
    /// `s` and whose card total is `n`. The whole grid is one flat `Vec<f64>` (no per-cell allocation);
    /// the moment slots let the consumer read `E[k_j]` per cell without a second pass.
    ///
    /// Built by folding one class at a time (`build`), so no configuration is ever materialized and the
    /// cost tracks the table size — polynomial in `N` — rather than the exponential number of configs.
    ///
    /// This is the moment-table form used by the [`CountState`] oracle; production uses the leaner,
    /// incrementally-depletable [`CountW`] instead, so this is `#[cfg(test)]`.
    struct CountDp {
        s_min: i16,
        s_max: i16,
        n_max: u16,
        n_span: usize,
        width: usize,
        data: Vec<f64>,
    }

    impl CountDp {
        /// Flat index of slot 0 of cell `(s, n)`. Moment slot `1 + j` is at the returned index `+ 1 + j`.
        fn cell(&self, s: i16, n: u16) -> usize {
            let row = (s - self.s_min) as usize;
            (row * self.n_span + n as usize) * self.width
        }

        fn build(classes: &CountClasses) -> Self {
            let m = classes.values.len();
            let width = 1 + m;
            let n_max: u16 = classes.sizes.iter().sum();
            let n_span = n_max as usize + 1;
            // A class contributes `v_j · k_j` for `k_j ∈ [0, M_j]`, so the running count lives in
            // `[Σ min(0, v_j M_j), Σ max(0, v_j M_j)]`. Partial folds stay within these global bounds.
            let s_min: i16 = classes
                .values
                .iter()
                .zip(&classes.sizes)
                .map(|(&v, &sz)| (v * sz as i16).min(0))
                .sum();
            let s_max: i16 = classes
                .values
                .iter()
                .zip(&classes.sizes)
                .map(|(&v, &sz)| (v * sz as i16).max(0))
                .sum();
            let s_span = (s_max - s_min + 1) as usize;

            let mut cur = Self {
                s_min,
                s_max,
                n_max,
                n_span,
                width,
                data: vec![0.0; s_span * n_span * width],
            };
            // Seed: the empty sub-shoe — count 0, 0 cards, weight 1, all moments 0.
            let seed = cur.cell(0, 0);
            cur.data[seed] = 1.0;

            for (i, (&v_i, &m_i)) in classes.values.iter().zip(&classes.sizes).enumerate() {
                // This class's binomial row `C(M_i, k)` depends only on `k`, not on the cell `(s, n)`;
                // hoist it out of the s/n loops so each of the `M_i + 1` values is computed once instead
                // of `s_span · n_span` times. Local and contiguous — no shared cache, stays lock-free.
                let binom: Vec<f64> = (0..=m_i).map(|k| choose(m_i, k)).collect();
                let mut next = vec![0.0; cur.data.len()];
                for s in s_min..=s_max {
                    for n in 0..=n_max {
                        let src = cur.cell(s, n);
                        let w = cur.data[src];
                        // No config reaches this cell yet, so its moments are 0 too — skip.
                        if w == 0.0 {
                            continue;
                        }
                        for k in 0..=m_i {
                            let b = binom[k as usize];
                            let dst = cur.cell(s + v_i * k as i16, n + k);
                            // Base weight W picks up this class's binomial factor.
                            next[dst] += w * b;
                            // Already-folded class moments are carried, scaled by the same factor.
                            // (Not-yet-folded classes have moment 0 here, so skipping them is moot.)
                            for j in 0..m {
                                if j != i {
                                    next[dst + 1 + j] += cur.data[src + 1 + j] * b;
                                }
                            }
                            // Class `i`'s own first moment is introduced now: weight · k · C(M_i, k).
                            next[dst + 1 + i] += w * k as f64 * b;
                        }
                    }
                }
                cur.data = next;
            }
            cur
        }
    }

    /// KO's pivot is `+4` regardless of deck count, so `external = 4 − internal`. A full,
    /// undealt shoe sits at internal `+4n` (its `full_shoe_count`) and external `starting_count`.
    #[test]
    fn ko_pivot_and_conversion() {
        for n in [1u8, 2, 6, 8] {
            assert_eq!(Ko::full_shoe_count(n), 4 * n as i16);
            assert_eq!(Ko::starting_count(n), 4 - 4 * n as i16);
            assert_eq!(Ko::pivot(n), 4, "KO pivot is +4 for every shoe size");

            // The undealt shoe: internal count = full_shoe_count, external = IRC.
            let internal_full = Ko::full_shoe_count(n);
            assert_eq!(
                Ko::internal_to_external(n, internal_full),
                Ko::starting_count(n)
            );

            // Round-trip across a sweep of running counts, both static and instance APIs.
            let state = CountState::from_decks::<Ko>(n);
            assert_eq!(
                state.running_count(),
                internal_full,
                "fresh shoe internal count"
            );
            for internal in -20i16..=20 {
                let external = Ko::internal_to_external(n, internal);
                assert_eq!(external, 4 - internal);
                assert_eq!(Ko::external_to_internal(n, external), internal);
                assert_eq!(state.internal_to_external(internal), external);
                assert_eq!(state.external_to_internal(external), internal);
            }
        }
    }

    /// Balanced systems pivot at zero: `external = −internal`, independent of deck count.
    #[test]
    fn balanced_system_pivots_at_zero() {
        struct HiLo;
        impl CountSystem for HiLo {
            fn map(card: &Card) -> i16 {
                match card {
                    Card::Ace | Card::Ten => -1,
                    Card::Pip(r) if *r <= 6 => 1,
                    Card::Pip(_) => 0, // 7, 8, 9
                }
            }
        }
        for n in [1u8, 6, 8] {
            assert_eq!(HiLo::full_shoe_count(n), 0);
            assert_eq!(HiLo::pivot(n), 0);
            assert_eq!(HiLo::internal_to_external(n, 7), -7);
            assert_eq!(HiLo::external_to_internal(n, -7), 7);
        }
    }

    /// A [`CountShoe`] that *knows* the whole deck — conditioned on the full deck's own count with the
    /// penetration pinned to the full size — must reproduce the finite [`CardCol`] solve exactly,
    /// because the only admissible config at every node is the exact remaining deck (count Eq shifts
    /// and `CardsRemaining` decrements lock-step with the draws). This pins the entire count plumbing
    /// (`draw`/`remove_hand`/`all_draw_probs`, the dealer recursion, the DP) against the trusted path.
    ///
    /// `#[ignore]`d in routine runs (~1min: a full no-split solve over three up-cards). The fast
    /// `count_w_*` unit tests already pin the DP primitive bit-for-bit; this is the heavier end-to-end
    /// integration check, run on demand with `--ignored`.
    #[test]
    #[ignore]
    fn count_shoe_matches_finite_deck_when_fully_known() {
        use crate::rules::Ruleset;
        use crate::shoe::CardCol;
        use crate::simulation::build_evs;

        // Splitting disabled to keep the 1-deck solve fast; the hit/stand/double DP and the full
        // dealer recursion are still exercised end-to-end.
        let rules = Ruleset {
            max_split_hands: 1,
            ..Ruleset::default()
        };
        let full = CardCol::from_decks(1).len() as u16;
        for up in [Card::Pip(6), Card::Ten, Card::Ace] {
            let finite = build_evs(CardCol::from_decks(1), up, &rules);
            let cshoe = CountShoe::new::<Ko>(
                1,
                CountCondition::Eq(Ko::full_shoe_count(1)),
                Penetration::CardsRemaining(full),
            );
            let counted = build_evs(cshoe, up, &rules);

            assert_eq!(finite.len(), counted.len(), "tree size differs");
            for (hand, (w_f, moves_f)) in &finite {
                let (w_c, moves_c) = counted.get(hand).expect("count tree missing a hand");
                assert!((w_f - w_c).abs() < 1e-9, "pooling weight differs");
                assert_eq!(moves_f.len(), moves_c.len(), "move set differs");
                for (mv, ev_f) in moves_f {
                    let ev_c = moves_c.get(mv).expect("count tree missing a move");
                    assert!((ev_f - ev_c).abs() < 1e-9, "EV differs by {}", ev_f - ev_c);
                }
            }
        }
    }

    /// `CountW.draw_dist` (the incremental primitive) must reproduce the eager enumeration oracle for
    /// the equality condition, the same bar [`dp_matches_enumeration`] holds the moment-table DP to.
    #[test]
    fn count_w_draw_dist_matches_oracle() {
        let val = std::array::from_fn(|r| Ko::map(&Card::from_rank_index(r)));
        let cw = CountW::build(val, CardCol::from_decks(1));
        let state = CountState::from_decks::<Ko>(1);
        for c in [-2i16, 0, 2, 4] {
            let oracle: Vec<f64> = state.all_draw_probs_given_c(c).map(|(_, p)| p).collect();
            let got = cw.draw_dist(CountCondition::Eq(c), Penetration::FlatPastPercent(25));
            for (r, (&o, g)) in oracle.iter().zip(got).enumerate() {
                assert!(
                    (o - g).abs() < 1e-9,
                    "rank {r}, count {c}: oracle {o} vs CountW {g}"
                );
            }
        }
    }

    /// Depleting the table by `remove_card` (deconvolution) must equal rebuilding it on the reduced
    /// deck — this is what lets the dealer recursion deplete incrementally instead of rebuilding.
    #[test]
    fn count_w_remove_card_matches_rebuild() {
        let val = std::array::from_fn(|r| Ko::map(&Card::from_rank_index(r)));
        let removed = [Card::Ten, Card::Pip(5), Card::Ace, Card::Ten];
        let mut cw = CountW::build(val, CardCol::from_decks(1));
        for rank in removed {
            cw.remove_card(&rank);
        }
        let fresh = CountW::build(val, CardCol::from_decks(1) - CardCol::from_hand(&removed));
        for c in [-3i16, 0, 3] {
            let a = cw.draw_dist(CountCondition::Eq(c), Penetration::FlatPastPercent(25));
            let b = fresh.draw_dist(CountCondition::Eq(c), Penetration::FlatPastPercent(25));
            for (r, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() < 1e-9,
                    "rank {r}, count {c}: incremental {x} vs rebuild {y}"
                );
            }
        }
    }

    /// Timing harness (run with `--release --ignored --nocapture`): how long a single count-conditioned
    /// column solve takes versus deck size, to locate the cost wall of the exact-dealer path.
    #[test]
    #[ignore]
    fn time_count_column() {
        use crate::rules::Ruleset;
        use crate::simulation::build_evs;
        use std::time::Instant;
        for n in [1u8, 2] {
            let t = Ko::full_shoe_count(n) / 2;
            for (label, max_split) in [("no-split", 1u8), ("split", 4u8)] {
                let rules = Ruleset {
                    max_split_hands: max_split,
                    ..Ruleset::default()
                };
                let shoe = CountShoe::new::<Ko>(
                    n,
                    CountCondition::Eq(t),
                    Penetration::FlatPastPercent(25),
                );
                let start = Instant::now();
                let tree = build_evs(shoe, Card::Ten, &rules);
                eprintln!(
                    "n={n} {label}: build_evs {:?}, tree {} hands",
                    start.elapsed(),
                    tree.len()
                );
            }
        }
    }

    /// End-to-end behavioral check (run with `--release --ignored --nocapture`): conditioning on a
    /// higher KO running count should push a stiff hand (hard 16 vs ten) from Hit toward Stand — high
    /// count ⇒ low cards already gone ⇒ the next card is more likely a bust card. Proves the count
    /// actually flows through `build_evs` and shifts the decision.
    #[test]
    #[ignore]
    fn count_shifts_strategy() {
        use crate::hand::Move;
        use crate::rules::Ruleset;
        use crate::simulation::build_evs;

        let rules = Ruleset::default();
        let up = Card::Ten;
        let hand16 = CardCol::from_hand(&[Card::Ten, Card::Pip(6)]);
        for ext in [-10i16, 0, 10] {
            let shoe = CountShoe::from_external::<Ko>(
                1,
                ext,
                CountCmp::Eq,
                Penetration::FlatPastPercent(25),
            );
            let tree = build_evs(shoe, up, &rules);
            let (_, evs) = &tree[&hand16];
            let stand = evs[&Move::Stand];
            let hit = evs[&Move::Hit];
            eprintln!(
                "KO RC {ext:+}: 16vT  stand {stand:+.4}  hit {hit:+.4}  -> {}",
                if stand > hit { "STAND" } else { "HIT" }
            );
        }
    }

    /// The count-conditioned player edge must move the *right* way: a higher KO running count
    /// (player-favourable, low cards gone, deck ten/ace rich) must *raise* the edge. This pins the
    /// fix for the inverted-edge bug — the two-card root occurrence weight that the edge integrates
    /// over was the untilted hypergeometric scan-weight (count-favoured naturals/20s under-weighted),
    /// so the edge fell as the count rose. Routing the seed weight through `Shoe::hand_prob` (the
    /// count-tilted occurrence probability) restores the monotonicity.
    ///
    /// `#[ignore]`d in routine runs: a no-split edge pass over all ten up-cards at three counts on a
    /// 1-deck count shoe (~30s). The cheap `count_w_*`/conversion unit tests run by default; this is
    /// the heavier end-to-end guard, run on demand with `--ignored`.
    #[test]
    #[ignore]
    fn count_edge_rises_with_running_count() {
        use crate::rules::Ruleset;
        use crate::simulation::{build_evs, edge_term};

        // No split keeps the pass fast; the edge sign/monotonicity is a hit/stand/double property.
        let rules = Ruleset {
            max_split_hands: 1,
            ..Ruleset::default()
        };
        let edge_at = |ext: i16| -> f64 {
            let shoe = CountShoe::from_external::<Ko>(
                1,
                ext,
                CountCmp::Eq,
                Penetration::FlatPastPercent(25),
            );
            shoe.all_draw_probs()
                .collect::<Vec<_>>()
                .into_iter()
                .map(|(up, p_up)| p_up * edge_term(&build_evs(shoe.clone(), up, &rules)).value())
                .sum()
        };
        let low = edge_at(-10);
        let mid = edge_at(0);
        let high = edge_at(10);
        // Monotone increasing in the player's running count.
        assert!(
            low < mid && mid < high,
            "edge must rise with running count: RC-10 {low:+.4}, RC0 {mid:+.4}, RC+10 {high:+.4}"
        );
        // A strongly favourable count is a genuine player advantage (the Monte-Carlo true-mixture
        // mean over the matching subset population lands near +0.06; the sign is the load-bearing
        // claim here, not the magnitude).
        assert!(
            high > 0.0,
            "a strongly positive count must give the player an edge, got {high:+.4}"
        );
    }

    /// The DP draw distribution must match the eager enumeration reference bit-for-bit (up to FP).
    #[test]
    fn dp_matches_enumeration() {
        let state = CountState::from_decks::<Ko>(1);
        for c in [-2i16, 0, 2, 4] {
            let oracle: Vec<f64> = state.all_draw_probs_given_c(c).map(|(_, p)| p).collect();
            let dp = state.draw_probs_given_count(c);
            for (r, (&o, d)) in oracle.iter().zip(dp).enumerate() {
                assert!(
                    (o - d).abs() < 1e-9,
                    "rank {r}, count {c}: enumeration {o} vs dp {d}",
                );
            }
        }
    }
}
