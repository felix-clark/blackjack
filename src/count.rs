//! Definitions and systems of counting
//!
//! NOTE: I think that we might be able to do each "count" independently if we focus on the
//! "pre-deal" count, i.e. the count before the player's initial hand and the dealer's card are
//! shown. The realistic count would include the up-cards as well, so building a count-dependent
//! strategy table from this would need to look across multiple "pre-deal" EV charts to yield the
//! results for a given post-deal count. It's complicated by the fact that, to get precise results,
//! we need to track both the few exactly-known up-cards that impact the total count, as well as a
//! total count that marginalizes over all other possibilities with that constraint.

use crate::{
    card::Card,
    shoe::{CardCol, N_RANKS, Shoe},
};

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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
    // /// Construct this system for the given number of decks
    // fn for_decks(n: u8) -> Self {
    //     todo!()
    // }

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
    /// The owned incremental weight table for the current pool; depleted in place on `draw`.
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
}

impl PartialEq for CountShoe {
    fn eq(&self, other: &Self) -> bool {
        self.dp.deck == other.dp.deck
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
        self.dp.deck.hash(h);
        self.dp.value_of_rank.hash(h);
        self.cond.hash(h);
        self.pen.hash(h);
        self.mean_field.hash(h);
        self.mf_scale.hash(h);
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
        let internal = S::external_to_internal(n_decks, external);
        let cond = match cmp {
            CountCmp::Eq => CountCondition::Eq(internal),
            CountCmp::Ge => CountCondition::Le(internal),
            CountCmp::Le => CountCondition::Ge(internal),
        };
        Self::new::<S>(n_decks, cond, pen)
    }

    fn from_parts(
        dp: CountW,
        cond: CountCondition,
        pen: Penetration,
        dist: [f64; N_RANKS],
    ) -> Self {
        Self {
            dp,
            cond,
            pen,
            dist,
            mean_field: false,
            mf_scale: 0,
        }
    }

    /// The split solver's [`Shoe::for_split`] target: a finite deck whose composition is the
    /// count-tilted *expected remaining pool*, built at high resolution (`scale` units per real card)
    /// so the sub-card tilt survives. Draws deplete it `scale` units at a time (depletion exactly
    /// `1/n`) and read probabilities straight off it, so the split sub-solve gets both the count tilt
    /// and finite depletion at finite-deck speed (the `CountW` table is left unused). `dp.deck` is
    /// repurposed as that scaled mean-field composition.
    fn mean_field_view(&self) -> Self {
        let n = self.dp.deck.len() as u16;
        let (comp, scale) = expected_composition(&self.dist, n);
        let mut next = self.clone();
        // In mean-field mode the `CountW` weight table is never read — draws come straight off the
        // tilted `dp.deck` composition (see `draw`/`draw_prob`/`all_draw_probs`/`remove_hand`). Drop it
        // so the many mean-field shoes the split solver caches (one per reachable composition, in
        // `dealer_cache`/`draw_cache`/`memo`) don't each retain a full `s_span × n_span` table. At 4+
        // decks that retention blew memory into the GBs and got the process SIGKILLed; the table is
        // ~300 KB there and thousands of distinct compositions are reached per split, times the caches
        // and the concurrent up-card columns.
        next.dp.w = Vec::new();
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
        self.dp.deck
    }

    /// Deplete the pool by one card *without* recomputing the cached distribution — used to batch a
    /// whole-hand removal before a single recompute. The condition shifts by the card's value (so the
    /// running-count constraint follows the card down the tree) and the penetration prior ages.
    fn deplete(&mut self, card: &Card) {
        let v = self.dp.value_of_rank[card.rank_index()];
        self.dp.remove_card(card);
        self.cond = self.cond.shifted(v);
        self.pen = self.pen.after_draw();
    }

    /// Rebuild the cached count-conditioned draw distribution for the current pool/condition.
    fn recompute(&mut self) {
        self.dist = self.dp.draw_dist(self.cond, self.pen);
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
    for r in 0..N_RANKS {
        col.add_n(Card::from_rank_index(r), counts[r]);
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
        // In both modes: yield only ranks present in `dp.deck` (matching `CardCol::all_draw_probs`) —
        // a depleted rank would otherwise make the player Hit DP look up an un-enumerable child. In
        // mean-field mode the weights are the finite composition; otherwise the count-tilted `dist`.
        let mean_field = self.mean_field;
        let dist = self.dist;
        let deck = self.dp.deck;
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
        hand.is_submultiset(&self.dp.deck)
    }

    /// The enumerator's per-rank supply is the pool count. The hypergeometric scan-weight is not the
    /// count-conditioned joint and is treated as a loose pooling weight (see `simulation.rs`); the
    /// count-correct probabilities come from [`draw_prob`](Self::draw_prob)/[`all_draw_probs`].
    fn rank_count(&self, rank: &Card) -> Option<u16> {
        Some(self.dp.deck.get_count(rank))
    }

    fn for_split(&self) -> Self {
        self.mean_field_view()
    }
}

/// Incremental count **weight** table: the base generating coefficients
///
/// ```text
///   W[s][n] = Σ_{configs: running count s, size n} ∏_r C(M_r, k_r)
/// ```
///
/// for the current pool, flattened over (running count `s`, remaining size `n`). Unlike the test-only `CountDp`
/// it stores *only* `W` — no moment slots — and supports **O(cells) single-card removal** by
/// deconvolution (dividing the generating polynomial by one `(1 + x^v y)` factor). So the dealer and
/// player recursions build it once per shoe and deplete it incrementally as cards are drawn, instead
/// of rebuilding the whole DP on every draw. The per-class first moment the draw distribution needs is
/// recovered on demand from the same deconvolution via the identity
///
/// ```text
///   T_j = M_j · x^{v_j} y · ( W / (1 + x^{v_j} y) ),   so  T_j[s][n] = M_j · H_j[s − v_j][n − 1]
/// ```
///
/// where `H_j` is `W` deconvolved by class `j`. The generating variable `x` tracks the count `s` and
/// `y` the size `n`; multiplying by `(1 + x^v y)` is "add one card of value `v`", dividing removes one.
#[derive(Clone)]
struct CountW {
    /// per-rank count value `v_r`, indexed like [`CardCol`].
    value_of_rank: [i16; N_RANKS],
    /// the current pool (per-rank `M_r`); a class size `M_j` is the sum of `M_r` over ranks sharing
    /// value `v_j`.
    deck: CardCol,
    s_min: i16,
    s_max: i16,
    /// the current pool size — the largest `n` that carries weight; shrinks by 1 per card removed.
    n_max: u16,
    n_span: usize,
    /// flat `[s][n]` table; `W[s][n]` lives at `(s − s_min) * n_span + n`.
    w: Vec<f64>,
}

impl CountW {
    fn at(&self, s: i16, n: u16) -> f64 {
        if s < self.s_min || s > self.s_max {
            return 0.0;
        }
        self.w[(s - self.s_min) as usize * self.n_span + n as usize]
    }

    /// Build the table for `deck` under the per-rank value map `value_of_rank`, folding the pool one
    /// rank's cards at a time (each card multiplies the polynomial by `(1 + x^{v} y)`).
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
        // Seed the empty sub-shoe: count 0, size 0, weight 1.
        w[(0 - s_min) as usize * n_span] = 1.0;

        let mut me = Self {
            value_of_rank,
            deck,
            s_min,
            s_max,
            n_max,
            n_span,
            w,
        };
        for r in 0..N_RANKS {
            let v = value_of_rank[r];
            for _ in 0..deck.get_count_i(r) {
                me.fold_in_card(v);
            }
        }
        me
    }

    /// Multiply the polynomial in place by `(1 + x^{v} y)` — i.e. add one card of value `v`.
    /// `W'[s][n] = W[s][n] + W[s − v][n − 1]`; iterating `n` downward keeps the `n − 1` term unmodified.
    fn fold_in_card(&mut self, v: i16) {
        for n in (1..=self.n_max).rev() {
            for s in self.s_min..=self.s_max {
                let prev = self.at(s - v, n - 1);
                if prev != 0.0 {
                    self.w[(s - self.s_min) as usize * self.n_span + n as usize] += prev;
                }
            }
        }
    }

    /// `W` deconvolved by one card of value `v`: `H = W / (1 + x^{v} y)`, so
    /// `H[s][n] = W[s][n] − H[s − v][n − 1]`. Iterating `n` upward makes the `n − 1` term already final.
    fn deconv(&self, v: i16) -> Vec<f64> {
        let mut h = vec![0.0; self.w.len()];
        for n in 0..=self.n_max {
            for s in self.s_min..=self.s_max {
                let idx = (s - self.s_min) as usize * self.n_span + n as usize;
                let sub = if n == 0 || s - v < self.s_min || s - v > self.s_max {
                    0.0
                } else {
                    h[(s - v - self.s_min) as usize * self.n_span + (n - 1) as usize]
                };
                h[idx] = self.w[idx] - sub;
            }
        }
        h
    }

    /// Remove one card of `rank` from the pool, depleting the table *in place* by deconvolution:
    /// `W[s][n] ← W[s][n] − H[s−v][n−1]` with `H` overwriting `W` as we sweep `n` upward (the `n−1`
    /// term is already final by the time it is read). O(cells), no allocation.
    fn remove_card(&mut self, rank: &Card) {
        let v = self.value_of_rank[rank.rank_index()];
        for n in 1..=self.n_max {
            for s in self.s_min..=self.s_max {
                let sub = self.at(s - v, n - 1); // already deconvolved (n-1 done this sweep)
                if sub != 0.0 {
                    self.w[(s - self.s_min) as usize * self.n_span + n as usize] -= sub;
                }
            }
        }
        self.deck = self.deck - CardCol::from_hand(&[*rank]);
        self.n_max -= 1;
    }

    /// Next-card draw distribution conditioned on `cond` over the running count, under penetration
    /// prior `pen`. Mirrors the test-only `CountState::draw_probs_where` exactly, but reads the per-class moment
    /// from a deconvolution of the maintained `W` rather than a separately-stored moment table.
    fn draw_dist(&self, cond: CountCondition, pen: Penetration) -> [f64; N_RANKS] {
        let big_n = self.n_max;
        // `1/C(N,n)` row (multivariate-hypergeometric normalizer), as in `draw_probs_where`.
        let inv_cn: Vec<f64> = (0..=big_n).map(|n| 1.0 / choose(big_n, n)).collect();

        // For each distinct value present, the scalar Σ_{s accepts} Σ_n pen·invCn/n · H_v[s−v][n−1].
        // `acc[r] = M_r · S_{v_r}` then normalized (the M_j in T_j cancels the within-class M_r/M_j).
        let mut acc = [0.0; N_RANKS];
        let mut seen_values: Vec<i16> = Vec::new();
        let mut scale_of_value: Vec<f64> = Vec::new();
        for r in 0..N_RANKS {
            let m_r = self.deck.get_count_i(r);
            if m_r == 0 {
                continue;
            }
            let v = self.value_of_rank[r];
            let scale = if let Some(pos) = seen_values.iter().position(|&u| u == v) {
                scale_of_value[pos]
            } else {
                let h = self.deconv(v);
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
                        let hval = h[(sh - self.s_min) as usize * self.n_span + (n - 1) as usize];
                        if hval != 0.0 {
                            s_v += w_n * inv_cn[n as usize] / n as f64 * hval;
                        }
                    }
                }
                seen_values.push(v);
                scale_of_value.push(s_v);
                s_v
            };
            acc[r] = m_r as f64 * scale;
        }
        let total: f64 = acc.iter().sum();
        if total > 0.0 {
            acc.iter_mut().for_each(|p| *p /= total);
        }
        acc
    }
}

/// Binomial coefficient C(n, k) as `f64`, multiplicative form. Pure and stateless — no shared
/// cache — so it's inherently thread-safe and lock-free (the previous `#[cached]` Pascal recursion
/// serialized every call, including each recursive descent, on one global mutex).
///
/// Evaluated as `∏_{i=1..=k} (n-k+i) / i`, multiplying before dividing so each partial product is
/// the exact integer binomial `C(n-k+i, i)` (no fractional intermediates). Symmetry `C(n,k)=C(n,n-k)`
/// picks the smaller leg to minimize iterations. Values beyond 2^53 (e.g. `choose(416, 208)`) are
/// approximate, and only ever used in ratios where the error largely cancels.
fn choose(n: u16, k: u16) -> f64 {
    if k > n {
        return 0.;
    }
    let k = k.min(n - k);
    (1..=k).fold(1.0, |acc, i| acc * (n - k + i) as f64 / i as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;
    use std::collections::HashMap;

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
                for r in 0..N_RANKS {
                    // `knums` is indexed by class (same order as `classes.values`/`sizes`), so the class
                    // index is the only mapping needed — no value→index lookup.
                    let j = classes.class_of_rank[r];
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

            let mut acc = [0.0; N_RANKS];
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
                    for r in 0..N_RANKS {
                        let j = classes.class_of_rank[r];
                        let m_j = classes.sizes[j];
                        if m_j == 0 {
                            continue;
                        }
                        let m_r = self.deck.get_count_i(r) as f64;
                        let t_j = dp.data[base + 1 + j]; // T_j[s][n] = Σ_configs k_j · ∏ C
                        acc[r] += cell_w * (t_j / n as f64) * (m_r / m_j as f64);
                    }
                }
            }

            // Each cell's contribution already forms a sub-distribution over ranks, so normalizing just
            // divides out the total accepted mass (and guards FP drift). Stays all-zero if the condition
            // is unreachable, rather than producing NaNs.
            let total: f64 = acc.iter().sum();
            if total > 0.0 {
                acc.iter_mut().for_each(|p| *p /= total);
            }
            acc
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
