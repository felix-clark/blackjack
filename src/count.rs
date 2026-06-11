//! Definitions and systems of counting
//!
//! NOTE: I think that we might be able to do each "count" independently if we focus on the
//! "pre-deal" count, i.e. the count before the player's initial hand and the dealer's card are
//! shown. The realistic count would include the up-cards as well, so building a count-dependent
//! strategy table from this would need to look across multiple "pre-deal" EV charts to yield the
//! results for a given post-deal count. It's complicated by the fact that, to get precise results,
//! we need to track both the few exactly-known up-cards that impact the total count, as well as a
//! total count that marginalizes over all other possibilities with that constraint.

use itertools::Itertools;
use std::collections::HashMap;

use crate::{
    card::Card,
    shoe::{CardCol, N_RANKS},
};

// NOTE: If this ends up not needing to be anything more than a mapping, we can ditch the trait
// formalism and just pass in an arbitrary function Card -> i8 to CountState.
pub(crate) trait CountSystem {
    // fn for_decks(n: u8) -> Self;

    /// The initial running rount. Zero for balanced counts, so the default is implemented.
    fn starting_count(n_decks: u8) -> i16 {
        0
    }

    /// The mapping from card to count.
    /// NOTE: i16 is used because the space is necessary for total counts, and it's easier to
    /// maintain the per-card counts as well.
    /// NOTE: We could also implement this as an array of i8s, of length 10, corresponding to the
    /// internal CardCol array. This would probably optimize.
    fn map(card: &Card) -> i16;
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

pub(crate) struct CountState {
    /// The unknown pool: the full shoe minus any exactly-known removed cards. Every pool size `M_j`
    /// and count probability below is taken over this.
    deck: CardCol,
    /// The counting system materialized as a per-rank value `v_r`, indexed like `CardCol`'s inner
    /// array (`rank_index`). This *is* the whole system; the class grouping, pool sizes, and total
    /// counts are all derived from it together with `deck` via [`count_classes`](Self::count_classes).
    value_of_rank: [i16; N_RANKS],
}

impl CountState {
    /// This state's per-rank count values grouped into classes (the DP-friendly representation).
    fn count_classes(&self) -> CountClasses {
        CountClasses::from_value_map(self.value_of_rank, &self.deck)
    }

    fn from_deck(deck: CardCol, count_fn: fn(&Card) -> i16) -> Self {
        // The system is deck-independent, so materialize it for every rank (not just ranks present
        // in `deck`): a rank that is momentarily absent still has a well-defined count value.
        let value_of_rank = std::array::from_fn(|r| count_fn(&Card::from_rank_index(r)));
        Self {
            deck,
            value_of_rank,
        }
    }

    pub fn from_decks(n: u8, count_fn: fn(&Card) -> i16) -> Self {
        Self::from_deck(CardCol::from_decks(n), count_fn)
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
                let m_r = self.deck.get_count(&Card::from_rank_index(r)) as f64;
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
                    let m_r = self.deck.get_count(&Card::from_rank_index(r)) as f64;
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
            sizes[class_of_rank[r]] += deck.get_count(&Card::from_rank_index(r));
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

    #[test]
    fn ko_placeholder() {
        let state_deck = CountState::from_decks(2, Ko::map);
        // NOTE: These are not the player's count; they are intenal counts for the deck. To finish,
        // we'll need conversion functions, and in the simulation we'll also need to account for
        // up-cards.
        let draw_probs_0: Vec<(Card, f64)> = state_deck.all_draw_probs_given_c(0).collect();
        let draw_probs_m4: Vec<(Card, f64)> = state_deck.all_draw_probs_given_c(-4).collect();
        let draw_probs_4: Vec<(Card, f64)> = state_deck.all_draw_probs_given_c(4).collect();
        dbg!(draw_probs_0);
        dbg!(draw_probs_m4);
        dbg!(draw_probs_4);
        todo!()
    }

    /// The DP draw distribution must match the eager enumeration reference bit-for-bit (up to FP).
    #[test]
    fn dp_matches_enumeration() {
        let state = CountState::from_decks(1, Ko::map);
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
