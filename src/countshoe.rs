//! The count-conditioned solver: a [`Shoe`] whose draws are conditioned on a running/true count over
//! the unseen pool, plus the dynamic program ([`CountW`]) that computes that draw distribution.
//!
//! [`CountShoe`] is what the TUI swaps in for a plain [`CardCol`](crate::shoe::CardCol) once a count
//! condition is active: each draw both depletes the pool and shifts the carried
//! [`CountCondition`](crate::count::CountCondition), so the physical constraint "all visible cards
//! counted = C" is maintained as the tree descends. The expensive per-pool draw distribution is
//! produced by [`CountW`] — a normalized-probability DP over count classes — and memoized through a
//! shared [`DistCache`]. The counting *vocabulary* this conditions on (the systems, count kinds,
//! conditions and frames) lives in [`crate::count`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::card::Card;
use crate::count::*;
use crate::shoe::{CardCol, N_RANKS, Shoe};

/// Shared memo from a pool's **count-class composition** (and its [`CountFrame`]) to the per-rank
/// *value-scale* broadcast — the class-level half of the count-conditioned draw distribution (see
/// [`CountW::value_scales`]). The expensive deconvolution + `(s,n)`-sum depends on the pool *only*
/// through its class composition `(M_class)`, never the within-class rank breakdown; the breakdown
/// re-enters only as the cheap `m_r` multiplier when [`CountW::dist_from_scales`] reconstructs the
/// distribution. So keying on class composition collapses every pool that is class-identical but
/// rank-distinct (e.g. a dealer line that drew a 5 vs. a 6 — both KO +1) onto one deconvolution, on
/// top of the exact-pool dedup it replaces. Every [`CountShoe`] cloned within one solve shares the
/// same `Arc`. The frame is kept in the key because a single shared cache can serve several frames — a
/// count *band* (one deconvolution, many frames), and in particular a true-count band whose members
/// differ only by their visible offset (same condition, distinct `vis_*`), which the condition alone
/// would not distinguish.
///
/// The class composition is encoded as `[u16; N_RANKS]` with each rank holding its *whole class's*
/// pool count (`Σ_{r': v_{r'}=v_r} M_{r'}`, see [`CountW::class_counts`]) — a value identical across
/// class-equivalent pools and distinct otherwise, so it is a faithful class-composition key.
type DistCache = Arc<Mutex<HashMap<([u16; N_RANKS], CountFrame), [f64; N_RANKS]>>>;

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
    frame: CountFrame,
    pen: Penetration,
    /// The **root** pool's size and internal running count — the full shoe at construction, before any
    /// draw. The [`frame`](Self::frame) is anchored relative to this root, so reconstructing it from the
    /// (depleted) current [`pool`](Self::pool) needs the drawn totals `(root_internal − pool_internal,
    /// root_size − pool_size)`, which [`recompute`](Self::recompute) passes to
    /// [`CountW::value_scales`]. Fixed at construction; a solve never mixes shoes built from different
    /// roots, so these are excluded from `Eq`/`Hash` (solve-invariant, like the deck size itself).
    root_size: u16,
    root_internal: i16,
    /// The count-conditioned next-card distribution, cached so `draw_prob`/`all_draw_probs` are O(1);
    /// recomputed once per `draw` (which is when the pool changes).
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
    /// When this shoe is one member of a **count band** (a sweep solved together for the chart's
    /// count-index thresholds), the frames of *all* band members at the current pool, with
    /// `frame == band_frames[band_idx]`. Empty for an ordinary single-count solve. On a draw-distribution
    /// cache miss the whole band is filled from one deconvolution (see [`CountW::draw_dist_band`]), so
    /// sibling members sharing this shoe's `dist_cache` pay the expensive work once. Like `frame`, each
    /// entry is fixed (the draw bookkeeping lives in the root reconstruction, not the frame); excluded
    /// from `Eq`/`Hash` (it is a cache-warming hint, not part of the value the shoe represents — `frame`
    /// already pins that).
    band_frames: Vec<CountFrame>,
    /// This member's index into `band_frames` (0 when not banded).
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
            && self.frame == other.frame
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
        self.frame.hash(h);
        self.pen.hash(h);
        self.mean_field.hash(h);
        self.mf_scale.hash(h);
    }
}

/// The internal running count of `deck` under `value_of_rank` — `Σ_r v_r · M_r`, the count value of all
/// cards in the pool. Used to derive the cards-drawn totals for the root reconstruction (see
/// [`CountShoe::recompute`]).
fn internal_count(deck: &CardCol, value_of_rank: &[i16; N_RANKS]) -> i16 {
    (0..N_RANKS)
        .map(|r| value_of_rank[r] * deck.get_count_i(r) as i16)
        .sum()
}

impl CountShoe {
    /// A shoe of `n_decks` under counting system `S`, conditioned on the pure `cond` over the *internal*
    /// running count (no visible offset — the pre-round frame), with penetration prior `pen`. The caller
    /// converts a player's external running count to the internal condition via
    /// [`CountSystem::external_to_internal`]. Production builds shoes through [`framed`](Self::framed)
    /// (a `CountFrame`) or [`band_external`](Self::band_external); this pure-condition convenience is
    /// used only by the cross-check tests.
    #[cfg(test)]
    pub(crate) fn new<S: CountSystem>(n_decks: u8, cond: CountCondition, pen: Penetration) -> Self {
        Self::framed::<S>(n_decks, CountFrame::pre_round(cond), pen)
    }

    /// A shoe conditioned on a [`CountFrame`] — a constraint plus its decision-point visible offset (the
    /// Wizard-of-Odds per-frame entry, built by [`cond_for_frame`]). [`new`](Self::new) is this with a
    /// pre-round (offset-free) frame.
    pub(crate) fn framed<S: CountSystem>(n_decks: u8, frame: CountFrame, pen: Penetration) -> Self {
        let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
        let dp = CountW::build(value_of_rank, CardCol::from_decks(n_decks));
        let dist = dp.draw_dist(frame, pen);
        Self::from_parts(dp, frame, pen, dist)
    }

    /// Build from the player's *external* running count and a comparison, converting to the deck's
    /// internal count condition. The conversion inverts inequalities: a player count `≥ C` means the
    /// unseen pool's internal count is `≤ pivot − C` (more cards seen pushes the deck the other way).
    /// Test-only: production conditions through [`cond_for_frame`] + [`framed`](Self::framed).
    #[cfg(test)]
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

    /// Build from the player's entered count `value` and comparison, dispatched on the system's
    /// [`CountKind`] via [`cond_from_count`]: a running-count system reads `value` as the external
    /// running count (identical to [`from_external`](Self::from_external)), a true-count system reads it
    /// as the external true count and conditions on the joint `(s, n)` inequality. Test-only: production
    /// routes the selector through [`cond_for_frame`] + [`framed`](Self::framed) (per-frame, WoO).
    #[cfg(test)]
    pub(crate) fn from_count<S: CountSystem>(
        n_decks: u8,
        value: i16,
        cmp: CountCmp,
        pen: Penetration,
    ) -> Self {
        Self::new::<S>(n_decks, cond_from_count::<S>(n_decks, value, cmp), pen)
    }

    /// A **band** of shoes, one per [`CountFrame`] in `frames`, that share a single [`CountW`] build
    /// *and* one draw-distribution cache. The shared cache plus each member carrying the whole
    /// `band_frames` list means the expensive per-pool deconvolution is computed once for the entire band
    /// (every member's distribution extracted from one deconvolution — see [`CountW::draw_dist_band`])
    /// rather than once per member. Solve the first member to completion to warm the cache; the remaining
    /// members are then nearly free (pure cache hits). Used by the TUI to sweep the count axis for a
    /// column's count-index thresholds — for a running count the frames are distinct conditions
    /// (different counts); for a true count they may share a condition and differ only by visible offset.
    pub(crate) fn band<S: CountSystem>(
        n_decks: u8,
        frames: &[CountFrame],
        pen: Penetration,
    ) -> Vec<Self> {
        let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
        let dp = CountW::build(value_of_rank, CardCol::from_decks(n_decks));
        // `dp.deck` is the undepleted root, shared by every band member.
        let root_size = dp.deck.len() as u16;
        let root_internal = internal_count(&dp.deck, &dp.value_of_rank);
        let cache: DistCache = Arc::new(Mutex::new(HashMap::new()));
        frames
            .iter()
            .enumerate()
            .map(|(band_idx, &frame)| {
                let mut shoe = Self {
                    pool: dp.deck,
                    dp: dp.clone(),
                    frame,
                    pen,
                    root_size,
                    root_internal,
                    dist: [0.0; N_RANKS],
                    mean_field: false,
                    mf_scale: 0,
                    band_frames: frames.to_vec(),
                    band_idx,
                    dist_cache: cache.clone(),
                };
                shoe.recompute();
                shoe
            })
            .collect()
    }

    /// A [`band`](Self::band) over external **running** counts — the convenience the KO count-index
    /// sweep uses: one pre-round frame per count in `externals`, each the unseen-pool condition for that
    /// external count compared with `cmp` (see [`cond_from_external`]).
    pub(crate) fn band_external<S: CountSystem>(
        n_decks: u8,
        externals: &[i16],
        cmp: CountCmp,
        pen: Penetration,
    ) -> Vec<Self> {
        let frames: Vec<CountFrame> = externals
            .iter()
            .map(|&e| CountFrame::pre_round(cond_from_external::<S>(n_decks, e, cmp)))
            .collect();
        Self::band::<S>(n_decks, &frames, pen)
    }

    /// The occurrence distribution over **external** running counts for a full `n_decks` shoe under
    /// penetration prior `pen`: each `(c, P(c))` is how often a player who has counted down to the
    /// penetration point holds external running count `c`, summing to 1 and ascending in `c`. It is the
    /// marginal of the [`CountW`] count table over the prior's pool-size support (no count *condition* —
    /// this is the prior over which count you land on, not a conditional draw distribution). The
    /// count-index window is bounded by this: counts in its tails are practically never reached, so
    /// solving them is wasted (see [`super::tui`]'s index sweep). Cheap — one `CountW` build, no solving.
    pub(crate) fn external_count_distribution<S: CountSystem>(
        n_decks: u8,
        pen: Penetration,
    ) -> Vec<(i16, f64)> {
        let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
        let dp = CountW::build(value_of_rank, CardCol::from_decks(n_decks));
        let pivot = S::pivot(n_decks);
        let big_n = dp.n_max;
        // Marginalize the count table over the (flat) penetration prior on pool size `n`.
        let mut acc: Vec<f64> = vec![0.0; (dp.s_max - dp.s_min + 1) as usize];
        let mut total = 0.0;
        for n in 0..=big_n {
            let w = pen.weight(n, big_n);
            if w == 0.0 {
                continue;
            }
            for s in dp.s_min..=dp.s_max {
                let p = dp.at(s, n);
                if p != 0.0 {
                    acc[(s - dp.s_min) as usize] += w * p;
                    total += w * p;
                }
            }
        }
        // Convert internal count `s` to external `c = pivot − s`, normalize, drop zeros, sort ascending.
        let mut out: Vec<(i16, f64)> = Vec::new();
        for (i, &v) in acc.iter().enumerate() {
            if v > 0.0 {
                out.push((pivot - (dp.s_min + i as i16), v / total));
            }
        }
        out.sort_by_key(|&(c, _)| c);
        out
    }

    /// The occurrence distribution over **integer true counts** for a full `n_decks` shoe under
    /// penetration prior `pen`: each `(t, P(t))` is how often a deciding player holds a true count whose
    /// floor is `t`, summing to 1 and ascending in `t`. Floor-bucketed so the suffix sum `Σ_{t≥c} P(t)`
    /// is *exactly* `P(TC ≥ c)` for integer `c` — the tail mass the count-index slice search weights by
    /// (see [`super::tui::index`]). Pivot 0 (true-count systems are balanced), so `TC = −52·s/n`; the
    /// `n = 0` term has no true count and is skipped. Cheap — one `CountW` build, no solving.
    pub(crate) fn true_count_distribution<S: CountSystem>(
        n_decks: u8,
        pen: Penetration,
    ) -> Vec<(i16, f64)> {
        debug_assert_eq!(
            S::pivot(n_decks),
            0,
            "true-count occurrence assumes a balanced system"
        );
        let value_of_rank = std::array::from_fn(|r| S::map(&Card::from_rank_index(r)));
        let dp = CountW::build(value_of_rank, CardCol::from_decks(n_decks));
        let big_n = dp.n_max;
        let mut acc: HashMap<i16, f64> = HashMap::new();
        let mut total = 0.0;
        for n in 1..=big_n {
            let w = pen.weight(n, big_n);
            if w == 0.0 {
                continue;
            }
            for s in dp.s_min..=dp.s_max {
                let p = dp.at(s, n);
                if p == 0.0 {
                    continue;
                }
                // External true count = −(cards/deck)·s/n (pivot 0); floor-bucket to an integer.
                let tc = -(CARDS_PER_DECK as f64) * s as f64 / n as f64;
                *acc.entry(tc.floor() as i16).or_insert(0.0) += w * p;
                total += w * p;
            }
        }
        let mut out: Vec<(i16, f64)> = acc.into_iter().map(|(t, m)| (t, m / total)).collect();
        out.sort_by_key(|&(t, _)| t);
        out
    }

    fn from_parts(dp: CountW, frame: CountFrame, pen: Penetration, dist: [f64; N_RANKS]) -> Self {
        // `dp.deck` is still the undepleted root here (no draw has happened yet).
        let root_size = dp.deck.len() as u16;
        let root_internal = internal_count(&dp.deck, &dp.value_of_rank);
        Self {
            pool: dp.deck,
            dp,
            frame,
            pen,
            root_size,
            root_internal,
            dist,
            mean_field: false,
            mf_scale: 0,
            band_frames: Vec::new(),
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

    /// The `(drawn_sum, drawn_cards)` offsets [`recompute`](Self::recompute) derives from the root — the
    /// count-value sum and number of cards drawn since construction. Test accessor for
    /// `true_count_shoe_tracks_drawn_offsets`.
    #[cfg(test)]
    fn drawn(&self) -> (i16, u16) {
        (
            self.root_internal - internal_count(&self.pool, &self.dp.value_of_rank),
            self.root_size - self.pool.len() as u16,
        )
    }

    /// Deplete the pool by one card *without* recomputing the cached distribution — used to batch a
    /// whole-hand removal before a single recompute. Only the *current pool* and the penetration prior
    /// are updated here — both `O(1)`; the expensive `CountW` table is left untouched and synced lazily
    /// later (see [`sync_table`]). The count condition is **not** touched: it is anchored at the root and
    /// reconstructed against the depleted pool in [`recompute`](Self::recompute), so depletion only has
    /// to move the pool and age the prior.
    ///
    /// [`sync_table`]: Self::sync_table
    fn deplete(&mut self, card: &Card) {
        // One card off the current pool (per-rank saturating subtraction). NOT `remove_rank`, which
        // clears the whole rank.
        self.pool = self.pool - CardCol::from_hand(&[*card]);
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
        // The frame is anchored relative to the root pool; reconstruct that root from the current
        // (depleted) pool by the cards drawn since construction. `value_scales` adds these to each table
        // `(s, n)` before testing the frame. Both are functions of the pool (hence of `class_counts`), so
        // the cache key `(class_counts, frame)` still fully determines the result — no extra key
        // dimension (the frame's visible offset is fixed at construction, also part of the key).
        let drawn_cards = self.root_size - self.pool.len() as u16;
        let drawn_sum = self.root_internal - internal_count(&self.pool, &self.dp.value_of_rank);
        let class_counts = self.dp.class_counts_of(&self.pool);
        let key = (class_counts, self.frame);
        if let Some(&scales) = self.dist_cache.lock().unwrap().get(&key) {
            self.dist = self.dp.dist_from_scales_with(&self.pool, &scales);
            return;
        }
        // Miss: the deconvolution genuinely needs the table at the current pool.
        self.sync_table();
        if self.band_frames.len() <= 1 {
            let scales = self
                .dp
                .value_scales(self.frame, self.pen, drawn_sum, drawn_cards);
            self.dist_cache.lock().unwrap().insert(key, scales);
            self.dist = self.dp.dist_from_scales_with(&self.pool, &scales);
            return;
        }
        // Banded: one deconvolution serves the whole band. Fill every member's value scales for this
        // class composition so sibling solves sharing `dist_cache` hit them instead of re-deconvolving
        // (the band is filled atomically — all present or none — so the early cache check above is a
        // sound guard). Every member shares the same `(drawn_sum, drawn_cards)` — same pool.
        let scales = self
            .dp
            .value_scales_band(&self.band_frames, self.pen, drawn_sum, drawn_cards);
        {
            let mut cache = self.dist_cache.lock().unwrap();
            for (f, sc) in self.band_frames.iter().zip(scales.iter()) {
                cache.insert((class_counts, *f), *sc);
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
    /// `drawn_sum`/`drawn_cards` are the count-value sum and card count drawn since the **decision
    /// point** (`0` for a freshly built table). The condition is anchored at the root, so each table
    /// `(s, n)` is lifted to the root pair `(s + drawn_sum, n + drawn_cards)` before the condition is
    /// tested — see [`CountCondition::accepts`]. The penetration prior, by contrast, weights the
    /// *current* `n` (how deep we now are), exactly as before. For a running count this is identical to
    /// shifting the threshold by `−drawn_sum`, so KO is byte-for-byte unchanged; the `0, 0` call in
    /// [`draw_dist`](Self::draw_dist) makes the oracle cross-checks literal as well.
    ///
    /// [`dist_from_scales`]: Self::dist_from_scales
    fn value_scales(
        &self,
        frame: CountFrame,
        pen: Penetration,
        drawn_sum: i16,
        drawn_cards: u16,
    ) -> [f64; N_RANKS] {
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
                    let sh = s - v;
                    if sh < self.s_min || sh > self.s_max {
                        continue;
                    }
                    for n in 1..=big_n {
                        // Lift this table `(s, n)` to the root pair and test the frame there. The
                        // acceptance is per-`(s, n)`: a true-count frame admits `s` only at the pool
                        // sizes whose decision-point true count clears the cutoff, so the filter lives
                        // inside the `n` loop (a running-count frame ignores `n`, so KO is unchanged).
                        if !frame.accepts((s + drawn_sum) as i32, (n + drawn_cards) as i32) {
                            continue;
                        }
                        let w_n = pen.weight(n, big_n);
                        if w_n == 0.0 {
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

    /// Next-card draw distribution conditioned on `frame` over the running count, under penetration
    /// prior `pen`. The class-level [`value_scales`](Self::value_scales) (cached across class-equivalent
    /// pools) folded together with the rank-level `m_r` by [`dist_from_scales`](Self::dist_from_scales);
    /// kept as a single entry point for the oracle cross-checks.
    fn draw_dist(&self, frame: CountFrame, pen: Penetration) -> [f64; N_RANKS] {
        // No draws since this table was built — the frame is tested literally on the table `(s, n)`.
        self.dist_from_scales(&self.value_scales(frame, pen, 0, 0))
    }

    /// [`value_scales`] for a whole **band** of frames at once, sharing the expensive work.
    ///
    /// The frame only enters through the `frame.accepts(s, n)` filter; the `O(cells)` deconvolution
    /// `p_v` and the per-`(s, n)` term `w_n · p_v` are both frame-*independent*. So for a band of frames
    /// over the *same* pool we deconvolve each present value once, then distribute each `(s, n)` term to
    /// whichever band members admit it. Returns one value-scale broadcast per `frame`, in order — equal
    /// to calling [`value_scales`] per frame (same terms, only the summation regrouped per-`(s, n)` so
    /// float order differs in the last bits), at a fraction of the cost (a running-count band admits `s`
    /// independently of `n`, recovering the one-sum-per-`s` cost). This is what lets the solver sweep a
    /// count band (the chart's count-index thresholds) while paying the dominant deconvolution only once.
    ///
    /// [`value_scales`]: Self::value_scales
    fn value_scales_band(
        &self,
        frames: &[CountFrame],
        pen: Penetration,
        drawn_sum: i16,
        drawn_cards: u16,
    ) -> Vec<[f64; N_RANKS]> {
        let big_n = self.n_max;
        let mut outs = vec![[0.0; N_RANKS]; frames.len()];
        let mut seen_values: Vec<i16> = Vec::new();
        // Per distinct present value, the `s_v` scale for each band member (parallel to `frames`).
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
                let mut sv = vec![0.0; frames.len()];
                for s in self.s_min..=self.s_max {
                    let sh = s - v;
                    if sh < self.s_min || sh > self.s_max {
                        continue;
                    }
                    // The deconvolution `p_v` and the `(w_n · pval)` term are frame-independent; the
                    // *admission* is not (a true-count frame admits a given `s` only at some pool sizes
                    // `n`), so each term is distributed per-`(s, n)` to whichever members accept it. For
                    // a running-count band `accepts` ignores `n`, so this reduces to the old "one
                    // `n`-sum per `s`, handed to every admitting member" — same terms, only the
                    // summation regrouped (last-bit float differences against per-frame `value_scales`).
                    for n in 1..=big_n {
                        let w_n = pen.weight(n, big_n);
                        if w_n == 0.0 {
                            continue;
                        }
                        let pval = p_v[(sh - self.s_min) as usize * self.n_span + (n - 1) as usize];
                        if pval == 0.0 {
                            continue;
                        }
                        let term = w_n * pval;
                        for (ci, frame) in frames.iter().enumerate() {
                            if frame.accepts((s + drawn_sum) as i32, (n + drawn_cards) as i32) {
                                sv[ci] += term;
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
    fn draw_dist_band(&self, frames: &[CountFrame], pen: Penetration) -> Vec<[f64; N_RANKS]> {
        self.value_scales_band(frames, pen, 0, 0)
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
                    let dist = cw.draw_dist(CountFrame::pre_round(cond), pen);
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
        let frames: Vec<CountFrame> = conds.iter().map(|&c| CountFrame::pre_round(c)).collect();
        for n in [1u8, 2u8] {
            let mut cw = CountW::build(val, CardCol::from_decks(n));
            for step in 0..5 {
                let band = cw.draw_dist_band(&frames, pen);
                for (i, &cond) in conds.iter().enumerate() {
                    let single = cw.draw_dist(CountFrame::pre_round(cond), pen);
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
            let frames: Vec<CountFrame> = externals
                .iter()
                .map(|&e| CountFrame::pre_round(cond_from_external::<Ko>(n, e, CountCmp::Eq)))
                .collect();
            let band = CountShoe::band::<Ko>(n, &frames, pen);
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
            const KIND: CountKind = CountKind::TrueCount;
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

    /// The production [`HiLo`] is a balanced true-count system; KO is a running-count one. The engine
    /// branches on [`CountSystem::KIND`], so pin both classifications (and Hi-Lo's balance, which the
    /// true-count math assumes).
    #[test]
    fn count_kinds_are_classified() {
        assert_eq!(HiLo::KIND, CountKind::TrueCount);
        assert_eq!(Ko::KIND, CountKind::Running);
        for n in [1u8, 2, 6, 8] {
            assert_eq!(
                HiLo::full_shoe_count(n),
                0,
                "a true-count system must be balanced"
            );
            assert_eq!(HiLo::pivot(n), 0);
        }
    }

    /// [`cond_from_count`] dispatches on the system kind: a running system delegates to
    /// [`cond_from_external`] (inverting the inequality into internal-count space); a true-count system
    /// builds the `True*` joint condition directly (no inversion — already in the player's TC frame).
    /// True counts are inequality-only.
    #[test]
    fn cond_from_count_dispatches_on_kind() {
        // Running (KO): identical to the running-count path.
        for cmp in [CountCmp::Eq, CountCmp::Ge, CountCmp::Le] {
            assert_eq!(
                cond_from_count::<Ko>(6, 3, cmp),
                cond_from_external::<Ko>(6, 3, cmp),
            );
        }
        // True count (Hi-Lo): the half-unit cutoff carried verbatim (no inversion). `cond_from_count` is
        // the pre-round pure condition; the decision-point visible offset is `cond_for_frame`'s job.
        assert_eq!(
            cond_from_count::<HiLo>(6, 2, CountCmp::Ge),
            CountCondition::TrueGe { cutoff: 2 },
        );
        assert_eq!(
            cond_from_count::<HiLo>(6, -1, CountCmp::Le),
            CountCondition::TrueLe { cutoff: -1 },
        );
    }

    /// [`cond_for_frame`] is the Wizard-of-Odds per-frame entry. For a running count it reproduces the
    /// `external − map(U) − k` shift the chart/index merge has always used; for a true count it sets the
    /// visible offset `vis_sum = map(U) + k` (and the caller's `vis_cards`) so the TC is anchored at the
    /// decision point. `vis = (0, 0)` (an up-card and `k` that cancel, with no visible cards) recovers
    /// the pre-round [`cond_from_count`].
    #[test]
    fn cond_for_frame_matches_running_shift_and_true_offset() {
        // Running (KO): a pre-round frame whose condition is the explicit external shift.
        for cmp in [CountCmp::Ge, CountCmp::Le] {
            for &up in &[Card::Pip(6), Card::Ten, Card::Ace] {
                for k in -2i16..=2 {
                    assert_eq!(
                        cond_for_frame::<Ko>(6, 5, cmp, up, k, 3),
                        CountFrame::pre_round(cond_from_external::<Ko>(
                            6,
                            5 - Ko::map(&up) - k,
                            cmp
                        )),
                        "KO frame must equal the external shift (up={up:?} k={k})",
                    );
                }
            }
        }
        // True count (Hi-Lo): the pure half-unit cutoff paired with the decision-point visible offset.
        assert_eq!(
            cond_for_frame::<HiLo>(6, 3, CountCmp::Ge, Card::Ten, 1, 3),
            CountFrame {
                cond: CountCondition::TrueGe { cutoff: 3 },
                vis_sum: HiLo::map(&Card::Ten) + 1, // map(T) = −1 ⇒ 0
                vis_cards: 3,
            },
        );
    }

    /// Exact equality on a true count is a measure-zero event over the `(s, n)` lattice, so it is
    /// rejected at construction.
    #[test]
    #[should_panic(expected = "inequality-only")]
    fn true_count_eq_is_rejected() {
        let _ = cond_from_count::<HiLo>(6, 2, CountCmp::Eq);
    }

    /// [`CountFrame::accepts`] matches the floating-point **decision-point** true count
    /// `TC = −52·(s − vis_sum)/(n − vis_cards)` against the half-unit cutoff `cutoff/2` (away from exact
    /// boundaries, where the cross-multiplied integer form is the authority and float rounding could
    /// disagree). The visible offset is applied here, before the pure condition is tested — so the sweep
    /// ranges over the cutoff, the reconstructed root pair, *and* the visible offset. A non-positive
    /// decision-point size admits nothing.
    #[test]
    fn true_count_accepts_matches_float_reference() {
        for cutoff in -3i16..=3 {
            for (vis_sum, vis_cards) in [(0i16, 0u16), (1, 3), (-2, 3), (0, 1)] {
                let ge = CountFrame {
                    cond: CountCondition::TrueGe { cutoff },
                    vis_sum,
                    vis_cards,
                };
                let le = CountFrame {
                    cond: CountCondition::TrueLe { cutoff },
                    vis_sum,
                    vis_cards,
                };
                for s in -10i32..=10 {
                    for n in 1i32..=20 {
                        let n_dp = n - vis_cards as i32;
                        if n_dp <= 0 {
                            assert!(
                                !ge.accepts(s, n),
                                "Ge must admit nothing at non-positive dp size"
                            );
                            assert!(
                                !le.accepts(s, n),
                                "Le must admit nothing at non-positive dp size"
                            );
                            continue;
                        }
                        // Balanced ⇒ pivot 0; decision-point TC = −52·(s − vis_sum)/n_dp, compared to
                        // the half-unit cutoff cutoff/2.
                        let tc = -52.0 * (s - vis_sum as i32) as f64 / n_dp as f64;
                        let thr = cutoff as f64 / 2.0;
                        if (tc - thr).abs() <= 1e-9 {
                            continue; // on the boundary: integer form is exact, float may round
                        }
                        assert_eq!(
                            ge.accepts(s, n),
                            tc > thr,
                            "Ge s={s} n={n} vis=({vis_sum},{vis_cards})"
                        );
                        assert_eq!(
                            le.accepts(s, n),
                            tc < thr,
                            "Le s={s} n={n} vis=({vis_sum},{vis_cards})"
                        );
                    }
                }
            }
        }
    }

    /// A `CountShoe` must track the value-sum and card-count drawn since construction — the offsets
    /// [`CountShoe::recompute`] adds to each table `(s, n)` to reconstruct the **root** pair the
    /// condition is anchored on. This is what replaced the old per-draw `shifted` threading. Drawing a
    /// mixed hand and checking the offsets against a manual tally pins the reconstruction at the shoe
    /// boundary (the `accepts` predicate itself is pinned by `true_count_accepts_matches_float_reference`).
    #[test]
    fn true_count_shoe_tracks_drawn_offsets() {
        use crate::shoe::Shoe;
        let pen = Penetration::FlatPastPercent(25);
        let mut shoe = CountShoe::from_count::<HiLo>(2, 1, CountCmp::Ge, pen);
        assert_eq!(shoe.drawn(), (0, 0), "a fresh shoe has drawn nothing");
        let hand = [Card::Pip(5), Card::Ten, Card::Ace, Card::Pip(8)];
        let mut exp_sum = 0i16;
        for (i, c) in hand.iter().enumerate() {
            shoe.draw(c);
            exp_sum += HiLo::map(c); // +1, −1, −1, 0
            assert_eq!(
                shoe.drawn(),
                (exp_sum, (i + 1) as u16),
                "after drawing {c:?} (#{}) the offsets must equal the running tally",
                i + 1
            );
        }
    }

    /// End-to-end wiring of the true-count condition into the [`CountShoe`] draw distribution, pinned to
    /// the full deck (the only admissible pool, count 0): `TC ≥ 0` admits it and the distribution must
    /// equal the plain finite deck's; `TC ≥ 1` (cutoff `2` half-units) cannot be met by a count-0 pool,
    /// so no count-consistent mass remains. Mirrors [`count_shoe_matches_finite_deck_when_fully_known`]
    /// for the `True*` path, without a full solve.
    #[test]
    fn true_count_full_deck_pin_gates_on_cutoff() {
        let finite = CardCol::from_decks(1);
        let full = finite.len() as u16;

        let admit =
            CountShoe::from_count::<HiLo>(1, 0, CountCmp::Ge, Penetration::CardsRemaining(full));
        for r in 0..N_RANKS {
            let card = Card::from_rank_index(r);
            let (got, want) = (admit.draw_prob(&card), finite.draw_prob(&card));
            assert!(
                (got - want).abs() < 1e-12,
                "TC≥0 over the full deck must match the finite deck: rank {r} {got} vs {want}"
            );
        }

        let reject =
            CountShoe::from_count::<HiLo>(1, 2, CountCmp::Ge, Penetration::CardsRemaining(full));
        let mass: f64 = (0..N_RANKS)
            .map(|r| reject.draw_prob(&Card::from_rank_index(r)))
            .sum();
        assert_eq!(mass, 0.0, "TC≥1 over a count-0 pool must leave no mass");
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

    /// The count-frame fix for insurance, pinned to an exact value via full-deck equivalence.
    ///
    /// On a fresh, undealt shoe the running count is the IRC (`starting_count`). The dealer's exposed
    /// Ace is one card the player *sees and counts*, so the running count they hold at the insurance
    /// decision — the Wizard-of-Odds convention, where the entered count includes every visible card —
    /// is `IRC + map(Ace) = IRC - 1`. With one Ace gone the hole card is drawn from the full deck minus
    /// that Ace, so `P(hole = Ten)` is the exact finite value `16n / (52n - 1)`.
    ///
    /// The corrected recipe ([`ShoeChoice::insurance`](crate::tui)) anchors at `entered - map(Ace)`
    /// before drawing the Ace, so the post-draw distribution lands at the entered count with the Ace
    /// removed: pinning the pool to the full size (`CardsRemaining`) makes the only admissible config the
    /// exact full-deck-minus-Ace, reproducing that finite value bit-for-bit. The old (no-offset) recipe
    /// would instead condition on `entered`, i.e. `IRC - 1`, an internal count no full shoe can have.
    #[test]
    fn insurance_count_frame_matches_finite_deck() {
        use crate::shoe::CardCol;
        for n in [1u8, 2, 6] {
            let full = CardCol::from_decks(n).len() as u16;
            // Running count the player holds once the fresh-shoe Ace is exposed and counted.
            let entered = Ko::starting_count(n) + Ko::map(&Card::Ace);
            // Corrected recipe: anchor at the pre-Ace count, then draw the Ace (which shifts it back).
            let c_in = entered - Ko::map(&Card::Ace);
            let mut shoe = CountShoe::new::<Ko>(
                n,
                cond_from_external::<Ko>(n, c_in, CountCmp::Eq),
                Penetration::CardsRemaining(full),
            );
            shoe.draw(&Card::Ace);
            let p_ten = shoe.draw_prob(&Card::Ten);
            let n = n as u32;
            let expected = (16 * n) as f64 / (52 * n - 1) as f64;
            assert!(
                (p_ten - expected).abs() < 1e-9,
                "n={n}: hole-Ten prob {p_ten} vs exact finite-deck {expected}"
            );
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
            let got = cw.draw_dist(
                CountFrame::pre_round(CountCondition::Eq(c)),
                Penetration::FlatPastPercent(25),
            );
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
            let pen = Penetration::FlatPastPercent(25);
            let a = cw.draw_dist(CountFrame::pre_round(CountCondition::Eq(c)), pen);
            let b = fresh.draw_dist(CountFrame::pre_round(CountCondition::Eq(c)), pen);
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
