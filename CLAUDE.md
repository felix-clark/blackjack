# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build: `cargo build`
- Run: `cargo run` (launches the TUI — entry point is `tui::run`, called from `src/main.rs`)
- Check without building: `cargo check`
- Lint: `cargo clippy`
- Format: `cargo fmt`
- Tests: `cargo test` (regression `#[test]`s pin verified EVs/strategy cells, chiefly in `src/simulation.rs`, `src/split.rs`, `src/reach.rs`, and `src/countshoe.rs`; shared test scaffolding is in `src/test_support.rs`; some compute functions also self-check with `assert!`). The count-conditioned solver's slowest cross-checks are marked `#[ignore]` — run them with `cargo test --release -- --ignored`.

This is edition 2024 Rust. External dependencies are kept deliberately few: `ratatui` (the TUI, confined to `src/tui/`), `serde` + `bincode` (used only by `src/diskcache.rs` for on-disk persistence; the cached types `derive` the serde traits), `rand` (sampling in the training drill), and `itertools`. The solver engine and all non-TUI/non-cache modules remain standard-library-only in spirit — including the parallelism (`par_map` is `std::thread::scope`, not `rayon`).

**Disk cache invalidation — read before changing the solver math.** `src/diskcache.rs` persists solved `Column`s and `IndexReport`s to `$XDG_CACHE_HOME/blackjack/v{SCHEMA_VERSION}/` keyed by `(up-card, shoe, ruleset[, count])`. The key captures the *inputs* (including the full `Ruleset`, split precision and all) but **not** the solver algorithm itself: if you change anything that alters computed EVs/strategy without changing those input types — a bug fix, a precision change, a payoff tweak — a stale cached value will be served silently. **Bump `SCHEMA_VERSION` in `src/diskcache.rs` whenever a cached type's layout *or* the EV computation changes** (it lives in the path, so a bump transparently orphans every older file). When validating the solver itself, set `BLACKJACK_NO_CACHE` to force clean re-solves.

## Architecture

The project computes **optimal blackjack basic strategy and per-hand expected values (EVs)** by exact enumeration over a finite shoe (or an infinite deck), optionally conditioned on a card-counting system, and presents it through a `ratatui` TUI. There is no game *loop* in the solver — it computes EVs/strategy outright — but the **training** tab does play hand-by-hand rounds against the live shoe.

### Core data model (`src/card.rs`, `src/shoe.rs`)

- `Card` (`src/card.rs`) is `Ace | Pip(2..=9) | Ten`. Tens and all face cards collapse into `Ten`; rank is all that matters in blackjack, so suits and 10/J/Q/K distinctions do not exist. `Card::hard()` gives the always-low value (Ace = 1). `Card::rank_index()` (Ace→0 … Ten→9) and `Card::from_rank_index()` map ranks to/from the dense array index used everywhere; `Card::ALL` is the canonical ten-rank array in that order (use it instead of re-spelling the literal).
- `CardCol` (`src/shoe.rs`) is a **dense** multiset — `[u16; 10]` of per-rank counts indexed by `rank_index` — and is the single representation for **both a hand and a shoe**. It is `Copy` with derived `Hash`/`Eq` (an absent rank is simply a `0`, so there is no zero-vs-missing ambiguity). Key helpers: `hard_count()` (aces low), `best_count()` (one ace promoted to 11 when it fits), `has_ace()`, `is_nat21()`, `is_submultiset()`, `highest_rank()`, `remove_rank()`, `from_decks(n)`, `try_from("9A")` for terse hand literals. `Sub` is per-rank saturating subtraction (multiset difference).
- `Shoe` trait (`src/shoe.rs`) abstracts the draw source: `draw`, `draw_prob`, `all_draw_probs`, plus the lazy `weighted_partitions` enumerator. Implemented by `CardCol` (finite, depletes on draw), `InfiniteDeck` (fixed 1/13 per rank, 4/13 for Ten; draws are no-ops), and `CountShoe` (the count-conditioned source, see below).
- The **partition enumerator** is the combinatorial heart: it enumerates every multiset of cards summing to a given hard total, each weighted by its multivariate hypergeometric probability. `WeightedPartitions` is a lazy stack machine over a dense `CardCol`. The weight bookkeeping is subtle; `check_hg_weights`/`check_hg_norm_weights` cross-check it against `choose(n,k)` terms via `assert!`.

### Hand/move vocabulary and rules (`src/hand.rs`, `src/rules.rs`)

- `HandState` (`Bust | Soft(n) | Hard(n) | Natural`) is the collapsed state a concrete hand maps to (`HandState::from(&CardCol)`); `Move` is the player option set (`Stand | Hit | Double | Split | Surrender`); `HandCategory` is the chart row a hand is filed under, with `categorize`/`pair_rank` routing a concrete `CardCol` into it.
- `Ruleset` (`src/rules.rs`) is the rule knobs threaded through the whole solver — HS17, DAS, the split-accuracy budget (`split_cards`), the blackjack payout, and `PeekRule` (the dealer-peek + surrender axis, bundled so the invalid no-peek/late-surrender combination is unrepresentable). Pure data; no compute lives here.

### Solver pipeline (`src/simulation.rs`, `src/dealer.rs`, `src/split.rs`, `src/reach.rs`)

- `src/dealer.rs` owns the dealer side: `DealerHand` (`[u8; 10]` memo key carrying the hit/stand/natural policy) and `dealer_outcome_probs`, the exact memoized enumeration of the dealer's outcome distribution over `DealerOutcome` (`Bust | Total(n) | Natural`). Memoizing on the dealer hand collapses the factorial of draw orders to the distinct reachable hands.
- `src/simulation.rs` is the **solver engine**. `build_evs(shoe, up_card, rules)` is the main driver: it removes the up card, then iterates player totals **21 down to 2** so `Hit` EV can read the already-computed EV of the post-hit hand (a dynamic program over the partition lattice), computing `Stand`/`Hit`/`Double`/`Surrender` EVs against the conditional dealer distribution. Result: `HashMap<CardCol, (weight, HashMap<Move, f64>)>`. `Basis` bundles the dealer-outcome and player-draw distributions plus the peek conditioning shared with the split solver; `resolve_ev` is the terminal payoff table. `summarize_evs`/`summarize_cells` collapse the per-exact-hand tree into the per-category strategy chart. `edge_term`/`bs_value_tree` compute the overall player edge and the basic-strategy-vs-optimal gap shown in the footer.
- `src/split.rs` is the pair-split machinery (`split_move_ev` → `SplitSolver`), a budget recursion over the split arms resting on the same `Basis`. **Split solves dominate runtime** (~98% on a multi-deck shoe) and are independent of the DP, so they're computed up front in parallel (`pair_split_evs_for` → `par_map`, a small `std::thread::scope` work-stealing helper sized to `available_parallelism()`); the DP then looks each pair's split EV up. The TUI runs the ten up-card columns concurrently, so splits from every column share the cores.
- `src/reach.rs` is the **game-time weighting** the live chart pools by, replacing the combinatorial scan-weight. `reach_weights(...)` runs a forward pass over the optimal policy to get each composition's *decision-reaching probability* — how often a deciding player actually holds it — fixing the cross-size weighting bias. Mass flows only on a `Hit` (a `Double` is terminal); split arms are folded back in. `summarize_cells` decides each cell's headline on its two-card decision population, flags composition-dependence, and feeds the popup breakdown. The correction shifts EVs in the 4th decimal and essentially never flips a cell (it's `0` on the infinite deck); `combinatoric_weights`/`summarize_evs` are retained as the regression baseline.

### Count conditioning (`src/count.rs`, `src/countshoe.rs`)

The card-counting dimension is split into **vocabulary** and **engine**:

- `src/count.rs` — what a count *is*: the `CountSystem` trait and its `Ko`/`HiLo` impls, `CountKind` (running vs. true-count family), the runtime `CountSystemId` + `dispatch_system!` seam, and the `CountCondition`/`CountFrame`/`Penetration`/`CountCmp` types plus the `cond_from_*`/`cond_for_frame` constructors that turn a player's entered count into a constraint over the unseen pool. Pure definitions; the rest of the codebase imports from here.
- `src/countshoe.rs` — the count-conditioned **solver**: `CountShoe`, a `Shoe` whose draws are conditioned on a `CountCondition` (each draw both depletes the pool and shifts the carried condition), backed by `CountW`, a normalized-probability dynamic program over count classes memoized through a shared `DistCache`. Main tree and dealer are exact; splits use a mean-field tilt. This file also holds the independent enumeration **oracle** (`CountState`/`CountDp`) the DP is cross-checked against in tests.

**Ace-Five running-count caveat (8-deck footgun).** The displayed "key count" (the lowest count whose edge crosses non-negative) is the player edge *marginalized over penetration* (`COUNT_PENETRATION = FlatPastPercent(25)`, a flat prior over 0–75% dealt). For a **balanced count read as a raw running count** — i.e. Ace-Five, where there is no true-count division — a fixed running count corresponds to a *higher* true count the deeper the shoe is dealt, so the edge at a fixed RC climbs with penetration and the marginal overstates the early-shoe edge. Quantified by the `ace_five_edge_by_penetration` measurement in `src/tui/index.rs` (`Penetration::CardsRemaining` pins the pool size; computes via `build_evs`/`edge_term` directly, **not** `load_or_build_frame`, whose disk key omits penetration). Per-round flat-bet edge (%) at fixed RC, early→deep penetration:

| decks | RC | 15% dealt | 50% dealt | 75% dealt | marginal (key count) |
|---|---|---|---|---|---|
| 6 | +4 | +0.033 | +0.392 | +1.298 | +0.383 (key ≈ RC +3) |
| 8 | +4 | **−0.139** | +0.138 | +0.825 | +0.121 (key ≈ RC +3/+4) |

So **6-deck is safe** (edge is ~break-even at the key count even at the start of the shoe), but **8-deck backfires**: at the marginal key count the first ~third of the shoe is still ~−0.14% EV, and you need ~RC +6 to be positive across all penetrations. Mitigation that keeps the count division-free: only ramp the bet past ~half the shoe (or add ~+2 to the trigger on 8 decks). This is intrinsic to running-count usage of a balanced count, not a solver bug — don't "fix" it in the math. The caveat is surfaced to users in the F1 count-description panel's notes (`CountSystemId::notes`).

### TUI (`src/tui/`)

The only place `ratatui` is used. Organised into top-level `Tab`s — **strategy** (the chart) and **training** (the drill) — with a documented module map at the top of `src/tui/mod.rs`:

- `config` — the solve configuration (`ShoeChoice`, `CountSetting`) and the per-column solve entry; this is where a plain `CardCol` is swapped for a `CountShoe` when a count is active.
- `column` — a solved up-card `Column` and the generic `solve_on`.
- `index` — the count-index subsystem (the running counts at which a cell's play flips), and the footer "+EV from count" key counts (the lowest count at which the player edge / insurance turn positive, found by per-band marginal differencing — `banded_crossing`).
- `training` — the training-tab model and game-simulation loop.
- `app` — the `App` state, the async solve lifecycle (one worker thread per up-card, results streamed over an `mpsc` channel and tagged with a monotonic `epoch` so a rules/deck/count change discards stale results), and the event loop.
- `input` — keyboard input and the modal field editors. **F1** opens the read-only *count-description* overlay (`Mode::CountInfo`) — a summary of the selected counting system (card tags, IRC/pivot/balance, the +EV and insurance key counts, and `CountSystemId::notes` usage caveats). It is the first of an intended F-key family of chart info overlays; currently a toggle (a true "hold to peek" would need terminal keyboard-enhancement release events).
- `render` — all drawing (incl. `render_count_info`, the F1 panel).

### Persistence (`src/diskcache.rs`)

The one place the project leaves its std-only-plus-`ratatui` discipline (caching needs serialization). Best-effort and side-channel: every operation degrades to a recompute on any I/O or decode error, so a missing/corrupt/older cache never breaks correctness, only speed. See the disk-cache invalidation warning above.

### Legacy and conventions

- `src/legacy.rs` is intentional reference code — earlier, slower implementations kept for cross-reference (and to keep some `CardCol` helpers un-strippable). It is `#[allow(unused)]` and not on the hot path.
- When extending, mirror the existing convention: **enumerate exactly** (any approximate fast path must be opt-in, never a silent substitution), assert distributions sum to 1 within ~1e-12, and validate new combinatorial weights against the hypergeometric cross-checks. Reference EV/strategy numbers come from wizardofodds.com.
