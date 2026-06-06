# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build: `cargo build`
- Run: `cargo run` (the binary's `main` is currently a scratchpad of assertions and demos, not a real CLI)
- Check without building: `cargo check`
- Lint: `cargo clippy`
- Format: `cargo fmt`
- Tests: `cargo test` (no `#[test]` functions exist yet; correctness is currently enforced by `assert!`/`dbg!` inside the compute functions and in `main`)

This is edition 2024 Rust with **no external dependencies** (the standard library only).

## Architecture

The project computes **optimal blackjack basic strategy and per-hand expected values (EVs)** by exact enumeration over a finite shoe (or an infinite deck). There is no game loop or player interaction — it's a solver.

### Core data model (`src/card.rs`, `src/shoe.rs`)

- `Card` (`src/card.rs`) is `Ace | Pip(2..=9) | Ten`. Tens and all face cards collapse into `Ten`; rank is all that matters in blackjack, so suits and 10/J/Q/K distinctions do not exist. `Card::hard()` gives the always-low value (Ace = 1). `Card::rank_index()` (Ace→0 … Ten→9) and `Card::from_rank_index()` map ranks to/from the dense array index used everywhere.
- `CardCol` (`src/shoe.rs`) is a **dense** multiset — `[u16; 10]` of per-rank counts indexed by `rank_index` — and is the single representation for **both a hand and a shoe**. It is `Copy` with derived `Hash`/`Eq` (an absent rank is simply a `0`, so there is no zero-vs-missing ambiguity to work around). Key helpers: `hard_count()` (aces low), `best_count()` (one ace promoted to 11 when it fits), `has_ace()`, `is_nat21()`, `is_submultiset()`, `highest_rank()`, `remove_rank()`, `from_decks(n)`, `half_deck()`, `try_from("9A")` for terse hand literals. `Sub` is per-rank saturating subtraction (multiset difference). It was previously backed by the `counter` crate with hand-rolled `Hash`/`PartialEq`; that is gone.
- `Shoe` trait (`src/shoe.rs`) abstracts the draw source: `draw`, `draw_prob`, `all_draw_probs`. Implemented by `CardCol` (finite, depletes on draw) and `InfiniteDeck` (fixed 1/13 per rank, 4/13 for Ten, draws are no-ops).

### Solver pipeline (`src/main.rs`)

- `HandState` (`Bust | Soft(n) | Hard(n) | Natural`) and `DealerOutcome` (`Bust | Total(n) | Natural`) are the *collapsed* states used for strategy summaries and payoff resolution. `HandState::from(&CardCol)` is the canonical hand→state mapping.
- `dealer_hit(hand, hs17)` encodes the dealer's fixed drawing rule (hit/stand on soft 17 toggled by `hs17`).
- Dealer outcome distributions over `DealerOutcome` come in two equivalent implementations. `_dealer_outcome_probs(hand, shoe)` is the original generic (`impl Shoe`) reference, still used for the `InfiniteDeck` demo in `main`. `_dealer_outcome_probs_dense(hand, shoe)` is the production path used by `build_evs`: same exact enumeration, but on `Copy` primitives — a dense `DenseShoe` (`[u32; 10]` + cached `total`) and a `DealerHand` (`[u8; 10]`, also the memo key carrying the hit/stand/natural policy) — memoized on the dealer hand so the factorial of draw orders collapses to the distinct reachable hands. It is ~30x faster and verified bit-identical. `remove_nat21(...)` renormalizes after excluding dealer naturals (used when the dealer peeks for blackjack).
- The partition enumerator is the combinatorial heart: it enumerates every multiset of cards summing to a given hard total, each weighted by its multivariate hypergeometric probability. `_weighted_partitions(deck, hard_total, norm_offset)` is the eager reference; `weighted_partitions_iter(deck, hard_total)` is the lazy stack-machine version `build_evs` actually uses (a leaf's `weight_part` factors telescope to `N!`). Both operate on dense `CardCol`. The weight bookkeeping is subtle and partially exploratory — `check_hg_weights`/`check_hg_norm_weights` exist purely to cross-check these weights against `choose(n,k)` terms via `assert!`.
- `build_evs(shoe, up_card)` is the main driver: removes the up card, then iterates player totals **21 down to 2** so that `Hit` EV can look up the already-computed EV of the post-hit hand in `full_ev_tree` (dynamic programming over the partition lattice). For each player hand it computes `Stand` and `Hit` EVs against the conditional dealer distribution. Result: `HashMap<CardCol, (weight, HashMap<Move, f64>)>`.
- `resolve_ev(player_hand, dealer_outcome)` is the terminal payoff table (natural pays 1.5, pushes, busts, comparisons).
- `consolidate_strategy(ev_tree)` collapses the per-exact-hand EV tree into one best `Move` per `HandState` via weighted averaging across all concrete hands sharing a state.

### End goal

The target is a **TUI** for asking arbitrary probability questions about blackjack states. The user specifies:
- a hand or hand state (e.g. a concrete `9A` or an abstract `Hard 14`),
- the dealer's up-card,
- a ruleset,
- optionally a running/true count under a chosen card-counting system,

and the program displays the EV of each available player move. Alternative views are also planned, e.g. printing the full basic-strategy chart for a given ruleset (and optional count). The current solver (`build_evs` → `consolidate_strategy`) is the compute backend for that front end; `main` is throwaway scaffolding around it.

Implications for current work:
- The count dimension means draw probabilities must be conditionable on a known partial-deck composition — `Shoe`/`CardCol` are already the right seam for this, but a counting system is a *mapping from deck composition to a count value* and the reverse (count → adjusted draw distribution) is the new piece.
- Rules need to become first-class: the `Ruleset` struct (HS17, DAS, dealer check, with a `Default`) already exists but is unused — thread it through `dealer_hit`, `resolve_ev`, and the dealer-outcome logic instead of the current hardcoded `hs17 = true` / `dealer_checks_blackjack = true`.

### Status

The code is mid-development and intentionally annotated. Several things are stubbed or partial: `add_double_evs` is `unimplemented!()`, `Move::{Double, Split, Surrender}` exist but aren't yet scored, the (so far unused) `Ruleset` struct is the planned home for rule variants, and many `TODO`s flag where weights/iterators/allocations are meant to be revisited. There is no TUI layer yet. When extending, mirror the existing convention: enumerate exactly, assert distributions sum to 1 within ~1e-12, and validate new weights against the hypergeometric cross-checks. Reference EV/strategy numbers come from wizardofodds.com (linked in `main`).
