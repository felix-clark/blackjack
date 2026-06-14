//! The **training tab**: a hand-by-hand blackjack drill against the live shoe, for practising basic
//! strategy, count-indexed deviations, and the running count.
//!
//! This module owns the training *model* ([`Training`] and its supporting types) and the *game engine*.
//! The harness around it — tab switching, the event loop, key routing ([`super::input`]) and rendering
//! ([`super::render`]) — is wired up in the sibling modules; the round lifecycle lives here:
//!
//! - [`Training::deal`] — start a round; the opening cards are drawn up front but laid out one at a time
//!   (paced, like a real table) before naturals/peek are resolved.
//! - [`Training::player_move`] — apply a player action and route the round forward.
//! - [`Training::start_dealer`] — hand off to the dealer, whose hand then plays out one paced card at a
//!   time under the house rule (driven by [`Training::tick`]).
//! - [`Training::settle`] — resolve payouts against the dealer and record the round.
//! - [`Training::evaluate`] — grade the player's decision against the reference plays and the EV gap.
//!
//! These deliberately reuse the solver engine rather than reimplementing blackjack: the dealer draws via
//! [`DealerHand`] (the very type [`crate::simulation`] uses), payouts via [`resolve_ev`], and the
//! decision grading via [`build_evs_with_splits`] on the live shoe — so the trainer's "optimal" play is
//! guaranteed consistent with the strategy chart. The lower-level primitives ([`Training::draw`],
//! [`Training::reveal`], [`Training::reset_shoe`], the count quiz [`Training::submit_count`]) are the
//! shoe/count plumbing the round logic sits on.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use rand::rngs::SmallRng;

use crate::card::Card;
use crate::count::{CountSystem, Ko};
use crate::dealer::DealerHand;
use crate::diskcache;
use crate::hand::{HandState, Move, best_move, categorize, pair_rank};
use crate::rules::Ruleset;
use crate::shoe::{CardCol, InfiniteDeck, Shoe};
use crate::simulation::{build_evs_with_splits, pair_split_evs_for, resolve_ev};

use super::column::{ColumnSummary, solve_on};
use super::config::ShoeChoice;
use super::index::{IndexKey, IndexReport};

/// The pace of the opening deal: one card laid every [`DEAL_STEP`] (brisk, the way a dealer pitches the
/// round) and one *dealer-turn* card every [`DEALER_STEP`] (slower, more deliberate). Both are coarser
/// than the event loop's ~100ms poll, which is what advances them; spreading the cards out one at a time
/// lets the counter keep up instead of a whole hand appearing at once.
const DEAL_STEP: Duration = Duration::from_millis(300);
const DEALER_STEP: Duration = Duration::from_millis(650);

/// Where a round currently sits in its lifecycle.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Phase {
    /// Between rounds — waiting for the player to deal.
    Ready,
    /// The opening deal is being laid out one card per [`DEAL_STEP`] tick (player, up, player, hole) —
    /// drawn up front but revealed in sequence; the player can't act until it finishes.
    Dealing,
    /// The player is acting on `hands[active]`.
    Player,
    /// The player has finished; the dealer is drawing — one card per [`DEALER_STEP`] tick (the hole
    /// flips first, then any draws), so the count is followable rather than dumped all at once.
    Dealer,
    /// The round is resolved and its per-hand outcomes are on screen.
    Settled,
}

/// How a single player hand finished, for display and payout. Constructed by [`Training::settle`].
#[derive(Clone, Copy, PartialEq)]
pub(super) enum HandResult {
    /// A two-card natural that beat the dealer (pays the blackjack multiplier).
    Blackjack,
    Win,
    Push,
    Lose,
    Bust,
    Surrender,
}

impl HandResult {
    /// A short label for the result column.
    pub(super) fn label(self) -> &'static str {
        match self {
            HandResult::Blackjack => "blackjack",
            HandResult::Win => "win",
            HandResult::Push => "push",
            HandResult::Lose => "lose",
            HandResult::Bust => "bust",
            HandResult::Surrender => "surrender",
        }
    }
}

/// One seat the player is acting on. A round starts with a single hand; a split adds more (the harness
/// model already carries the `from_split` flag and a per-hand bet so the simulation can grow this `Vec`).
///
/// `bet`/`from_split`/`done` are the round-progression fields the (stubbed) simulation drives; they are
/// modelled here so the engine has a complete hand record to fill.
#[derive(Clone)]
pub(super) struct TrainHand {
    /// Cards in the order they were dealt, for a natural left-to-right display.
    pub(super) cards: Vec<Card>,
    /// The wager on this hand (a double or split scales/duplicates it).
    pub(super) bet: f64,
    pub(super) doubled: bool,
    /// Whether this hand came from splitting a pair (so it is not eligible for a blackjack payout).
    pub(super) from_split: bool,
    /// Whether this hand was surrendered
    pub(super) surrendered: bool,
    /// The player has finished acting on this hand (stood, doubled, busted, or hit to 21).
    pub(super) done: bool,
    /// Filled at [`Training::settle`].
    pub(super) result: Option<HandResult>,
    /// Net units won (`+`) or lost (`−`) on this hand, filled at [`Training::settle`].
    pub(super) net: f64,
}

impl TrainHand {
    pub(super) fn new(bet: f64) -> Self {
        Self {
            cards: Vec::new(),
            bet,
            doubled: false,
            from_split: false,
            surrendered: false,
            done: false,
            result: None,
            net: 0.0,
        }
    }

    /// The hand as a rank multiset (the form the solver and [`HandState`] consume). Used by the
    /// simulation seam (e.g. [`Training::evaluate`]) to look the hand up in the EV tree.
    pub(super) fn col(&self) -> CardCol {
        CardCol::from_hand(&self.cards)
    }

    /// The collapsed [`HandState`] this hand is *presented* as for play and payout. A two-card 21 that
    /// arose from a split is a plain soft 21, **not** a blackjack (it neither pays the bonus nor pushes a
    /// dealer natural), so it is demoted here — the one place [`HandState::from`]'s raw `Natural` is not
    /// what the round wants. Everything else is the literal collapsed total.
    pub(super) fn state(&self) -> HandState {
        let raw = HandState::from(&self.col());
        if self.from_split && raw == HandState::Natural {
            HandState::Soft(21)
        } else {
            raw
        }
    }
}

/// A graded player decision: the move the player chose versus the three reference plays, with the EV of
/// each. Produced by [`Training::evaluate`] and surfaced in the feedback panel.
///
/// The three references are the standard training yardsticks:
/// - **basic** — the count-independent basic-strategy play (the chart's headline move),
/// - **indexed** — the count-index play at the *true* running count (the deviation a counter makes),
/// - **optimal** — the exact, composition-dependent best play for this very shoe and hand.
pub(super) struct DecisionMark {
    pub(super) chosen: Move,
    pub(super) basic: Move,
    /// The count-index deviation at the current running count, or `None` until the (expensive,
    /// count-conditioned) index lookup is wired up — the harness renders it as "n/a" meanwhile.
    pub(super) indexed: Option<Move>,
    pub(super) optimal: Move,
    /// EV of the move actually chosen.
    pub(super) ev_chosen: f64,
    /// EV of the exact-optimal move (so `ev_chosen - ev_optimal` is the mistake cost, ≤ 0 gap).
    pub(super) ev_optimal: f64,
}

/// Running training scoreboard: decision accuracy against each reference, realised vs. theoretical EV,
/// and count-quiz accuracy. All cumulative over the session; the render layer turns these into rates.
#[derive(Default)]
pub(super) struct TrainStats {
    pub(super) rounds: u32,
    pub(super) decisions: u32,
    /// Decisions matching the basic-strategy / indexed / exact-optimal reference, respectively.
    pub(super) agree_basic: u32,
    pub(super) agree_indexed: u32,
    pub(super) agree_optimal: u32,
    /// Cumulative EV the player gave up versus exact-optimal play (≤ 0; the sum of per-decision gaps).
    pub(super) ev_gap: f64,
    /// Cumulative net units actually won/lost across settled rounds.
    pub(super) realized: f64,
    pub(super) count_quizzes: u32,
    pub(super) count_correct: u32,
}

/// The trainer's live draw source. Its two modes are exactly the two games a trainer can drill, and
/// they are what cleanly separate the **game loop** from the **counting machinery**:
///
/// - [`TrainShoe::Finite`] is a real `n`-deck shoe that depletes as cards come out and is reshuffled at
///   penetration — the counted game, where the running count and the index drills mean something.
/// - [`TrainShoe::Infinite`] is a non-depleting infinite deck (every rank at its fixed `1/13`, Ten at
///   `4/13`): a continuous basic-strategy drill. There is no finite composition to track, so no count,
///   no penetration, and no reshuffle — the counting machinery is simply absent, and the round logic
///   that sits on this shoe is identical to the finite game.
pub(super) enum TrainShoe {
    Finite { cards: CardCol, n_decks: u8 },
    Infinite,
}

impl TrainShoe {
    fn from_choice(choice: ShoeChoice) -> Self {
        match choice {
            ShoeChoice::Infinite => TrainShoe::Infinite,
            ShoeChoice::Decks(n) => TrainShoe::Finite {
                cards: CardCol::from_decks(n),
                n_decks: n,
            },
        }
    }

    /// The [`ShoeChoice`] this shoe corresponds to, so [`Training::sync_shoe`] can tell when the
    /// selection changed under it.
    fn choice(&self) -> ShoeChoice {
        match self {
            TrainShoe::Infinite => ShoeChoice::Infinite,
            TrainShoe::Finite { n_decks, .. } => ShoeChoice::Decks(*n_decks),
        }
    }

    fn is_finite(&self) -> bool {
        matches!(self, TrainShoe::Finite { .. })
    }

    /// Rebuild a finite shoe to full; a no-op on the infinite deck (which never depletes).
    fn reset(&mut self) {
        if let TrainShoe::Finite { cards, n_decks } = self {
            *cards = CardCol::from_decks(*n_decks);
        }
    }

    /// Whether a finite shoe has passed its dealing penetration (here: under 25% of cards remain) and
    /// should be reshuffled before the next deal. The infinite deck never needs one.
    // TODO: associate with a configurable penetration setting.
    fn needs_shuffle(&self) -> bool {
        match self {
            TrainShoe::Finite { cards, n_decks } => cards.len() * 4 < *n_decks as usize * 52,
            TrainShoe::Infinite => false,
        }
    }

    /// The number of decks still unseen (for the true-count conversion), or `None` on the infinite deck
    /// — which has no finite composition and so no penetration to report.
    fn decks_remaining(&self) -> Option<f64> {
        match self {
            TrainShoe::Finite { cards, .. } => Some(cards.len() as f64 / 52.0),
            TrainShoe::Infinite => None,
        }
    }

    /// Draw one card using the given RNG.
    fn draw(&mut self, rng: &mut impl Rng) -> Card {
        match self {
            TrainShoe::Finite { cards, .. } => cards.draw_rand(rng),
            TrainShoe::Infinite => InfiniteDeck {}.draw_rand(rng),
        }
    }
}

/// The training-tab state: the live shoe, the in-progress round, the count-quiz overlay, and the
/// session scoreboard.
pub(super) struct Training {
    /// The trainer's live draw source: a finite, counted, depleting shoe or a non-depleting infinite
    /// deck (a pure basic-strategy drill — see [`TrainShoe`]).
    pub(super) shoe: TrainShoe,
    /// The *true* KO external running count of every card revealed so far — the value the count quiz
    /// grades against. Reset to the system IRC on a reshuffle.
    pub(super) running_count: i16,
    pub(super) phase: Phase,
    /// The dealer's cards: `[up, hole, ..draws]`. The hole card (index 1) stays hidden — and uncounted
    /// — until [`Training::dealer_step`] flips it on the first paced dealer tick (see [`hole_down`]).
    ///
    /// [`hole_down`]: Self::hole_down
    pub(super) dealer: Vec<Card>,
    pub(super) hands: Vec<TrainHand>,
    /// Index into `hands` of the hand currently being acted on.
    pub(super) active: usize,
    /// Whether the dealer's hole card is still face-down (so the render hides it and it stays out of the
    /// running count). Cleared the moment the paced dealer turn flips it in [`Training::dealer_step`].
    pub(super) hole_down: bool,
    /// The four opening cards, drawn up front (so the shoe is depleted and the hole is known for the peek)
    /// but laid into [`hands`](Self::hands)/[`dealer`](Self::dealer) one at a time by the paced
    /// [`deal_step`](Self::deal_step). Order: player, dealer up, player, dealer hole.
    opening: Vec<Card>,
    /// How many of [`opening`](Self::opening) have been laid out so far during [`Phase::Dealing`].
    opening_dealt: usize,
    /// When the next dealer card may be turned during [`Phase::Dealer`], or `None` when no step is
    /// pending. [`Training::tick`] fires the step once this instant passes; [`DEALER_STEP`] sets the pace.
    step_at: Option<Instant>,
    /// Whether the running-count quiz overlay is open.
    pub(super) entering_count: bool,
    /// The player's working count guess in the quiz overlay.
    pub(super) count_entry: i16,
    /// The most recent graded decision, shown in the feedback panel until the next one.
    pub(super) last_mark: Option<DecisionMark>,
    pub(super) stats: TrainStats,
    /// A one-line status/feedback message shown under the table.
    pub(super) message: String,
    /// Count-independent basic-strategy reference, solved lazily on the infinite deck (per up-card) and
    /// memoized for the session — see [`Training::evaluate`]. Invalidated when the ruleset changes.
    basic: HashMap<Card, ColumnSummary>,
    /// The ruleset `basic` was solved under; a change clears the cache.
    basic_rules: Ruleset,
    /// State for the harness's card draw.
    rng: SmallRng,
    /// Channel the background decision-grading workers ([`Training::spawn_eval`]) stream their finished
    /// [`EvalResult`]s back over. Mirrors the strategy tab's per-column solve plumbing in [`super::app`]:
    /// the grade is computed off the UI thread so a move never blocks on the ~split solve.
    eval_tx: Sender<EvalResult>,
    eval_rx: Receiver<EvalResult>,
    /// Monotonic id stamped on each grading job; orders [`last_mark`](Self::last_mark) updates and gates
    /// staleness against [`eval_valid_from`](Self::eval_valid_from).
    eval_seq: u64,
    /// Results with a `seq` below this were graded under a ruleset/deck the trainer has since left
    /// (a rules edit or shoe switch bumps it to the current `eval_seq`), so they are dropped on arrival.
    eval_valid_from: u64,
    /// The `seq` of the grade currently shown in `last_mark`; a later grade only overwrites it if its
    /// `seq` is at least this, so out-of-order completion never regresses the feedback panel.
    last_mark_seq: u64,
    /// Grading jobs still in flight, for the "grading…" hint while the worker runs.
    pending_evals: u32,
}

impl Training {
    pub(super) fn new(shoe: ShoeChoice) -> Self {
        let shoe = TrainShoe::from_choice(shoe);
        let running_count = initial_count(&shoe);
        let (eval_tx, eval_rx) = mpsc::channel();
        let rng = rand::make_rng();
        Self {
            shoe,
            running_count,
            phase: Phase::Ready,
            dealer: Vec::new(),
            hands: Vec::new(),
            active: 0,
            hole_down: false,
            step_at: None,
            opening: Vec::new(),
            opening_dealt: 0,
            entering_count: false,
            count_entry: 0,
            last_mark: None,
            stats: TrainStats::default(),
            message: "Press Enter to deal \u{00b7} n: guess the count \u{00b7} 1: strategy tab"
                .into(),
            basic: HashMap::new(),
            basic_rules: Ruleset::default(),
            rng,
            eval_tx,
            eval_rx,
            eval_seq: 0,
            eval_valid_from: 0,
            last_mark_seq: 0,
            pending_evals: 0,
        }
    }

    // ----- Provided primitives (not the simulation; the stubs below build on these) -----------------

    /// Rebuild the shoe to a full `n_decks` and reset the running count to the KO initial count. This is
    /// the reshuffle; the simulation should call it whenever penetration is reached (see
    /// [`Training::needs_shuffle`]).
    pub(super) fn reset_shoe(&mut self) {
        self.shoe.reset();
        // TODO: The counting system should derive from the setting in the strategy tab.
        self.running_count = initial_count(&self.shoe);
    }

    /// Whether the shoe has passed its dealing penetration and should be reshuffled before the next deal
    /// (always `false` on the infinite deck — see [`TrainShoe::needs_shuffle`]).
    pub(super) fn needs_shuffle(&self) -> bool {
        self.shoe.needs_shuffle()
    }

    /// Draw one card from the live shoe (uniformly at random from a finite shoe, which it depletes; at
    /// the fixed rank frequencies from the infinite deck, which it does not). This only depletes the
    /// shoe — it does **not** count the card; call [`Training::reveal`] when (and if) the card becomes
    /// visible to the player.
    pub(super) fn draw(&mut self) -> Card {
        self.shoe.draw(&mut self.rng)
    }

    /// Count a now-visible card into the running count. The hole card is drawn but *not* revealed until
    /// the dealer plays, so the player's running count tracks exactly the cards they can see. A no-op on
    /// the infinite deck, which has no count.
    pub(super) fn reveal(&mut self, card: Card) {
        if self.shoe.is_finite() {
            self.running_count += Ko::map(&card);
        }
    }

    /// The number of decks still unseen in the shoe (for converting the running count to a true count),
    /// or `None` on the infinite deck.
    pub(super) fn decks_remaining(&self) -> Option<f64> {
        self.shoe.decks_remaining()
    }

    /// Whether the live shoe is the finite, counted game (vs. the infinite-deck basic-strategy drill).
    /// The render/input layers gate the counting machinery — the count panel, the `n` quiz, the indexed
    /// reference — on this.
    pub(super) fn is_finite(&self) -> bool {
        self.shoe.is_finite()
    }

    /// Re-point the live shoe at the currently selected [`ShoeChoice`] if its deck count changed (e.g.
    /// the rules modal switched decks), abandoning any round dealt from the old shoe. Called when the
    /// training tab is entered.
    pub(super) fn sync_shoe(&mut self, shoe: ShoeChoice) {
        if shoe != self.shoe.choice() {
            self.reset_to(shoe);
        }
    }

    /// Re-initialise the trainer after the (shared) rules modal commits an edit: deal from a fresh shoe
    /// at the (possibly new) deck size and abandon any round, since a round in progress was dealt under
    /// the old rules. Unconditional — a rules change always resets the shoe (see [`reset_to`](Self::reset_to)),
    /// matching the request to re-shuffle on a rules *or* deck-size change.
    pub(super) fn on_rules_changed(&mut self, shoe: ShoeChoice) {
        self.reset_to(shoe);
    }

    /// Point the live shoe at `shoe` as a fresh full shoe, reset the running count, abandon any
    /// in-progress round, and drop any decision grade still in flight (it was computed for the old shoe).
    /// Shared by [`sync_shoe`](Self::sync_shoe) (deck switched) and [`on_rules_changed`](Self::on_rules_changed).
    fn reset_to(&mut self, shoe: ShoeChoice) {
        self.shoe = TrainShoe::from_choice(shoe);
        self.running_count = initial_count(&self.shoe);
        self.phase = Phase::Ready;
        self.hands.clear();
        self.dealer.clear();
        self.hole_down = false;
        self.step_at = None;
        self.opening_dealt = 0;
        self.last_mark = None;
        self.message =
            "Press Enter to deal \u{00b7} n: guess the count \u{00b7} 1: strategy tab".into();
        // Any grade still in flight was computed for the old shoe; drop it on arrival.
        self.eval_valid_from = self.eval_seq;
    }

    /// Grade the player's running-count guess against the true count and record it. Fully implemented —
    /// the count drill is part of the harness, not the blackjack simulation.
    pub(super) fn submit_count(&mut self) {
        self.stats.count_quizzes += 1;
        if self.count_entry == self.running_count {
            self.stats.count_correct += 1;
            self.message = format!("Count correct: RC {:+}", self.running_count);
        } else {
            self.message = format!(
                "Count was RC {:+} (you said {:+})",
                self.running_count, self.count_entry
            );
        }
        self.entering_count = false;
    }

    // ----- Round lifecycle --------------------------------------------------------------------------

    /// Start a new round: reshuffle at penetration, then draw the four opening cards (player, dealer up,
    /// player, dealer hole) up front — depleting the shoe now, and fixing the hole the dealer will peek
    /// against — and hand off to the paced [`Phase::Dealing`] that lays them out one at a time. The hole
    /// is dealt face-down and uncounted; the naturals/peek are resolved once the deal lands (see
    /// [`finish_opening_deal`](Self::finish_opening_deal)).
    pub(super) fn deal(&mut self, rules: &Ruleset) {
        if self.needs_shuffle() {
            self.reset_shoe();
        }
        self.hands = vec![TrainHand::new(1.0)];
        self.dealer.clear();
        self.hole_down = false;
        self.last_mark = None;
        self.active = 0;

        // Draw all four up front (so the hole exists for the peek and the shoe is depleted now), but reveal
        // them one per `DEAL_STEP` tick: player, dealer up, player, dealer hole.
        self.opening = vec![self.draw(), self.draw(), self.draw(), self.draw()];
        self.opening_dealt = 0;
        self.phase = Phase::Dealing;
        self.message = "Dealing\u{2026}".into();
        // Lay the first card immediately so the felt isn't momentarily empty; the rest follow on the timer.
        self.deal_step(rules);
    }

    /// Lay one opening card into its seat (revealing — and so counting — every card but the face-down
    /// hole), then arm the next step or, once all four are down, resolve naturals via
    /// [`finish_opening_deal`](Self::finish_opening_deal). Driven by [`tick`](Self::tick) in
    /// [`Phase::Dealing`].
    fn deal_step(&mut self, rules: &Ruleset) {
        let card = self.opening[self.opening_dealt];
        match self.opening_dealt {
            // Player's two cards, then the dealer up-card: all face-up and counted.
            0 | 2 => {
                self.reveal(card);
                self.hands[0].cards.push(card);
            }
            1 => {
                self.reveal(card);
                self.dealer.push(card);
            }
            // The hole: dealt face-down and uncounted until the dealer turn flips it.
            _ => {
                self.dealer.push(card);
                self.hole_down = true;
            }
        }
        self.opening_dealt += 1;
        if self.opening_dealt < self.opening.len() {
            self.arm_step(DEAL_STEP);
        } else {
            self.finish_opening_deal(rules);
        }
    }

    /// Resolve the just-dealt round: a peeked dealer blackjack (ten/ace up in a peek game) or a player
    /// natural hands straight to the dealer turn; otherwise it is the player's move. Reuses [`DealerHand`]
    /// for the natural check so "is this a blackjack" matches the solver exactly.
    fn finish_opening_deal(&mut self, rules: &Ruleset) {
        let up = self.dealer[0];
        // The dealer peeks only in a peek game with a ten-or-ace up-card; a peeked dealer blackjack ends
        // the round before the player can act (and before any double/split bet is laid). The hole is turned
        // over by the paced dealer turn (which then settles), not revealed inline.
        let dealer_natural = DealerHand::from_card_vec(&self.dealer).is_natural();
        let peeks = rules.peek.peeks() && matches!(up, Card::Ace | Card::Ten);
        if peeks && dealer_natural {
            self.start_dealer();
            return;
        }
        // A player natural stands pat: there is no decision, so go straight to the dealer (who, under no
        // peek, may still turn over a natural for a push).
        if self.hands[0].col().is_nat21() {
            self.hands[0].done = true;
            self.start_dealer();
            return;
        }

        self.phase = Phase::Player;
        self.message = "Your move \u{2014} legal actions are listed below.".into();
    }

    /// Apply a player action to the active hand, grade it, and route the round forward. An illegal move
    /// (the TUI offers every action key) is ignored with a message rather than mutating the hand.
    pub(super) fn player_move(&mut self, mv: Move, rules: &Ruleset) {
        if self.phase != Phase::Player || self.active >= self.hands.len() {
            return;
        }
        if !self.allowed_move(mv, rules) {
            self.message = format!("{} not allowed here", move_name(mv));
            return;
        }
        // Grade the decision against the pre-move state on a background worker; the result folds into
        // the scoreboard when it lands (see [`drain_evals`](Self::drain_evals)). The game advances
        // immediately below, so a move never blocks on the live solve.
        self.spawn_eval(mv, rules);

        match mv {
            Move::Hit => {
                let card = self.draw();
                self.reveal(card);
                self.hands[self.active].cards.push(card);
            }
            Move::Stand => self.hands[self.active].done = true,
            Move::Double => {
                let card = self.draw();
                self.reveal(card);
                let hand = &mut self.hands[self.active];
                hand.cards.push(card);
                hand.bet *= 2.0;
                hand.doubled = true;
                hand.done = true;
            }
            Move::Split => self.split_active(),
            Move::Surrender => {
                self.hands[self.active].surrendered = true;
                self.hands[self.active].done = true;
            }
        }
        self.advance();
    }

    /// Split the active pair into two hands, each re-drawing a card. Split aces get exactly one card and
    /// stand (the common rule). The new hand is inserted right after the active one so play proceeds
    /// left-to-right.
    fn split_active(&mut self) {
        let i = self.active;
        // A pair, so both cards share a rank; seed each arm with one of them.
        let rank = self.hands[i].cards[0];
        self.hands[i].cards.pop();
        self.hands[i].from_split = true;
        let mut new_hand = TrainHand::new(self.hands[i].bet);
        new_hand.from_split = true;
        new_hand.cards.push(rank);

        let c1 = self.draw();
        self.reveal(c1);
        self.hands[i].cards.push(c1);
        let c2 = self.draw();
        self.reveal(c2);
        new_hand.cards.push(c2);

        if rank == Card::Ace {
            self.hands[i].done = true;
            new_hand.done = true;
        }
        self.hands.insert(i + 1, new_hand);
    }

    /// Advance `active` past every finished hand; when none remain it is the dealer's turn. Called after
    /// each player action so a busted/standing/doubled hand hands off automatically.
    fn advance(&mut self) {
        while self.active < self.hands.len() && self.hand_finished(self.active) {
            self.active += 1;
        }
        if self.active >= self.hands.len() {
            self.start_dealer();
        }
    }

    /// Whether the player is done acting on `hands[i]`: it stood/doubled/surrendered, or its value forces
    /// it (a bust, a natural, or any 21 — there is nothing left to decide).
    fn hand_finished(&self, i: usize) -> bool {
        let h = &self.hands[i];
        h.done || h.surrendered || h.col().best_count() >= 21
    }

    /// Hand off to the dealer: enter [`Phase::Dealer`] and arm the first paced step. The hole stays
    /// face-down (and uncounted) until [`tick`](Self::tick) flips it one [`DEALER_STEP`] later; the draws
    /// then follow one per step. The actual card logic lives in [`dealer_step`](Self::dealer_step).
    fn start_dealer(&mut self) {
        self.phase = Phase::Dealer;
        self.arm_step(DEALER_STEP);
    }

    /// Advance any pending paced animation: called every event-loop tick. Once the
    /// [`step_at`](Self::step_at) deadline passes, turn exactly one card — an opening-deal card in
    /// [`Phase::Dealing`] (see [`deal_step`](Self::deal_step)) or a dealer card in [`Phase::Dealer`] (see
    /// [`dealer_step`](Self::dealer_step)). A no-op in every other phase.
    pub(super) fn tick(&mut self, rules: &Ruleset) {
        if self.step_at.is_none_or(|at| Instant::now() < at) {
            return;
        }
        self.step_at = None;
        match self.phase {
            Phase::Dealing => self.deal_step(rules),
            Phase::Dealer => self.dealer_step(rules),
            _ => {}
        }
    }

    /// Schedule the next paced step `delay` from now ([`DEAL_STEP`] for the opening deal, [`DEALER_STEP`]
    /// for the dealer turn).
    fn arm_step(&mut self, delay: Duration) {
        self.step_at = Some(Instant::now() + delay);
    }

    /// Whether the dealer still wants a card: a live player hand remains (against an all-bust /
    /// all-surrender table the dealer stands pat, as at a real table) and [`DealerHand::must_hit`] — the
    /// exact rule the solver's dealer uses — says hit on the current up+hole(+draws).
    fn dealer_wants_card(&self, rules: &Ruleset) -> bool {
        let any_live = self
            .hands
            .iter()
            .any(|h| !h.surrendered && !matches!(h.state(), HandState::Bust | HandState::Natural));
        any_live && DealerHand::from_card_vec(&self.dealer).must_hit(rules.hs17)
    }

    /// Turn one dealer card: the hole first (flipped and counted), then one drawn card per call. After the
    /// action it re-arms for the next step if the dealer wants another card, otherwise it settles. Each
    /// card is revealed (and so counted) exactly as it becomes visible, matching the live deal.
    fn dealer_step(&mut self, rules: &Ruleset) {
        if self.hole_down {
            self.hole_down = false;
            self.reveal(self.dealer[1]);
        } else {
            let card = self.draw();
            self.reveal(card);
            self.dealer.push(card);
        }
        if self.dealer_wants_card(rules) {
            self.arm_step(DEALER_STEP);
        } else {
            self.settle(rules);
        }
    }

    /// Resolve every hand against the dealer's final outcome, fill each hand's [`HandResult`]/`net` via
    /// the solver's [`resolve_ev`] payoff table, fold the round's net into [`TrainStats`], and settle.
    pub(super) fn settle(&mut self, rules: &Ruleset) {
        let bj_payout = rules.bj_payout.multiplier();
        let dealer_outcome = DealerHand::from_card_vec(&self.dealer).terminal_outcome();

        let mut round_net = 0.0;
        for hand in &mut self.hands {
            let state = hand.state();
            let (result, net) = if hand.surrendered {
                (HandResult::Surrender, -0.5 * hand.bet)
            } else {
                let ev = resolve_ev(state, dealer_outcome, bj_payout);
                let result = match state {
                    HandState::Bust => HandResult::Bust,
                    HandState::Natural if ev > 1e-9 => HandResult::Blackjack,
                    _ if ev > 1e-9 => HandResult::Win,
                    _ if ev < -1e-9 => HandResult::Lose,
                    _ => HandResult::Push,
                };
                (result, ev * hand.bet)
            };
            hand.result = Some(result);
            hand.net = net;
            round_net += net;
        }

        self.stats.rounds += 1;
        self.stats.realized += round_net;
        self.phase = Phase::Settled;
        self.message = format!("Round net {round_net:+.2} u \u{00b7} Enter to deal again");
    }

    /// Drain any finished background grades, folding each into the scoreboard (see
    /// [`fold_decision`](Self::fold_decision)). Stale results — graded under a ruleset/deck the trainer
    /// has since left — are dropped; a freshly-solved infinite-deck basic summary is memoized so the next
    /// decision on that up-card skips it. Called every event-loop tick from [`super::app`].
    pub(super) fn drain_evals(&mut self) {
        while let Ok(res) = self.eval_rx.try_recv() {
            self.pending_evals = self.pending_evals.saturating_sub(1);
            if res.seq < self.eval_valid_from {
                continue;
            }
            // A non-stale result was graded under the current ruleset (a rules change bumps
            // `eval_valid_from` past it), so its basic summary is safe to memoize.
            if let Some(summary) = res.basic_summary {
                self.basic.insert(res.up, summary);
            }
            if let Some(mark) = res.mark {
                self.fold_decision(res.seq, mark);
            }
        }
    }

    /// Whether a decision grade is still being computed in the background — the cue for the feedback
    /// panel's "grading…" hint.
    pub(super) fn grading(&self) -> bool {
        self.pending_evals > 0
    }

    /// Fold a graded decision into the running scoreboard and surface it in the feedback panel. `seq`
    /// orders the panel so an out-of-order completion never replaces a newer grade.
    fn fold_decision(&mut self, seq: u64, mark: DecisionMark) {
        self.stats.decisions += 1;
        if mark.chosen == mark.basic {
            self.stats.agree_basic += 1;
        }
        if mark.indexed == Some(mark.chosen) {
            self.stats.agree_indexed += 1;
        }
        if mark.chosen == mark.optimal {
            self.stats.agree_optimal += 1;
        }
        // The gap is the EV the choice gave up vs the exact-optimal play (≤ 0).
        self.stats.ev_gap += mark.ev_chosen - mark.ev_optimal;
        // Newest grade wins the feedback panel; cumulative stats above are order-independent.
        if seq >= self.last_mark_seq {
            self.last_mark_seq = seq;
            self.last_mark = Some(mark);
        }
    }

    /// Spawn a background worker to grade the player's move on the active hand against the reference
    /// plays, reusing the solver engine so the verdict matches the strategy chart exactly:
    /// - **optimal** (and the chosen/optimal EVs): the exact best play for *this* depleted shoe, from
    ///   [`build_evs_with_splits`] on the live composition — count-aware by construction; on the infinite
    ///   deck the (composition-independent) basic EVs already are exact-optimal, so no live solve runs;
    /// - **basic**: the count-independent infinite-deck headline, solved once per up-card and memoized;
    /// - **indexed**: the count-index deviation at the player's current running count, read from the
    ///   chart's **disk-cached** [`IndexReport`] (see [`indexed_move`](Self::indexed_move)) — `None` until
    ///   that up-card's index has been solved (it fills in the background while the strategy tab is open).
    ///
    /// All the heavy work ([`build_evs_with_splits`] and the cold infinite-deck basic solve) runs on the
    /// worker, so the move never blocks the UI; the cheap inputs (the index lookup, the reconstructed
    /// solver shoe, the memoized basic summary) are captured here and shipped to the pure
    /// [`run_eval`]. The finished grade folds back in via [`drain_evals`](Self::drain_evals).
    fn spawn_eval(&mut self, chosen: Move, rules: &Ruleset) {
        if self.dealer.len() < 2 || self.active >= self.hands.len() {
            return;
        }
        let up = self.dealer[0];
        let hand = self.hands[self.active].col();

        // A ruleset change invalidates the memoized basic summaries and any grade still in flight (it was
        // computed under the old rules), so drop both before issuing this job's `seq`.
        if self.basic_rules != *rules {
            self.basic.clear();
            self.basic_rules = *rules;
            self.eval_valid_from = self.eval_seq;
        }

        let job = EvalJob {
            seq: self.eval_seq,
            up,
            chosen,
            rules: *rules,
            // The moves legal on *this* (possibly multi-card) hand, so the basic-strategy reference is
            // judged among only the actions the player can actually take — a three-card 16 is graded on
            // Hit-vs-Stand, never against the two-card-only Surrender headline. Resolved here while we
            // still hold the live hand state.
            allowed: ALL_MOVES
                .into_iter()
                .filter(|&m| self.allowed_move(m, rules))
                .collect(),
            // Count-index deviation at the current running count (always `None` on the infinite deck —
            // see [`indexed_move`]); a cheap disk-cache lookup, resolved here on the UI thread.
            indexed: self.indexed_move(&hand, up, rules),
            // The round-start shoe for the finite live solve; `None` on the infinite deck, which routes
            // `run_eval` to the basic-EV (== optimal) path instead.
            finite_shoe: self.reconstruct_solver_shoe(&hand, up),
            // Reuse the memoized basic summary if we have it; otherwise the worker solves and echoes it.
            basic_summary: self.basic.get(&up).cloned(),
            hand,
        };
        self.eval_seq += 1;
        self.pending_evals += 1;

        let tx = self.eval_tx.clone();
        thread::spawn(move || {
            // Receiver gone (app exiting) is fine — just drop the result.
            let _ = tx.send(run_eval(job));
        });
    }

    /// Reconstruct the round-start shoe the live solve expects on a finite shoe: the live unseen deck,
    /// plus the (unseen) hole and the visible up-card and this hand's cards — which `build_evs` removes
    /// again, leaving it drawing from exactly the cards still hidden from the player (live shoe + hole).
    /// `None` on the infinite deck (whose optimal play is the composition-independent basic play).
    fn reconstruct_solver_shoe(&self, hand: &CardCol, up: Card) -> Option<CardCol> {
        let TrainShoe::Finite { cards, .. } = &self.shoe else {
            return None;
        };
        let hole = self.dealer[1];
        let mut solver_shoe = *cards;
        solver_shoe.insert(hole);
        solver_shoe.insert(up);
        for (card, n) in hand.iter() {
            solver_shoe.add_n(card, n);
        }
        Some(solver_shoe)
    }

    /// The count-index deviation move for the active hand at the player's current running count, read
    /// from the **disk-cached** [`IndexReport`] the strategy tab fills in the background (so this never
    /// triggers the heavy count-conditioned solve itself — a cold up-card simply returns `None`).
    ///
    /// The report's ladders are indexed by the external running count under the Wizard-of-Odds
    /// convention (the count *includes* the up-card and the hand), which is exactly what
    /// [`running_count`](Self::running_count) tracks. The headline ladder (`primary`) assumes a fresh
    /// two-card hand; if its play is a start-only move (double/split/surrender) the live hand can no
    /// longer make — already hit, or not a pair — we drop to the Hit/Stand `fallback` ladder, mirroring
    /// the popup's "if can't …" logic.
    fn indexed_move(&self, hand: &CardCol, up: Card, rules: &Ruleset) -> Option<Move> {
        // The index is a finite-shoe, count-conditioned object: the infinite deck has no count, so no
        // index deviation exists there. Keyed by the concrete deck count (the trainer's `n_decks`), it
        // shares the exact disk cache the strategy tab populates for a `Decks(n)` selection.
        let TrainShoe::Finite { n_decks, .. } = &self.shoe else {
            return None;
        };
        let key: IndexKey = (up, ShoeChoice::Decks(*n_decks), *rules);
        let report = diskcache::load::<_, IndexReport>("index", &key)?;
        let ci = report.cats.get(&categorize(hand))?;
        let primary = run_move(&ci.primary, self.running_count)?;
        if self.allowed_move(primary, rules) {
            Some(primary)
        } else {
            run_move(&ci.fallback, self.running_count)
        }
    }

    /// Whether `mv` is a legal action on the active hand. The TUI offers every action key, so this is the
    /// gate [`player_move`](Self::player_move) checks before applying one.
    pub(super) fn allowed_move(&self, mv: Move, rules: &Ruleset) -> bool {
        let hand = &self.hands[self.active];
        let n_cards = hand.cards.len();
        let n_hands = self.hands.len();
        match mv {
            // Cannot hit a doubled hand or one already at 21 (or bust, though such a hand is never active).
            Move::Hit => !hand.doubled && hand.col().best_count() < 21,
            Move::Stand => true,
            // Double on the first two cards; after a split it needs DAS.
            Move::Double => n_cards == 2 && (rules.das || n_hands == 1),
            // Split a fresh pair while under the hand cap.
            Move::Split => {
                n_cards == 2
                    && hand.cards[0] == hand.cards[1]
                    && n_hands < rules.max_split_hands as usize
            }
            // Surrender only the original two-card hand (not after a split or a hit), if the rules offer it.
            Move::Surrender => {
                rules.peek.surrender_offered() && n_hands == 1 && n_cards == 2 && !hand.from_split
            }
        }
    }
}

/// A decision-grading request shipped to a background worker by [`Training::spawn_eval`]. It carries the
/// inputs captured at decision time so [`run_eval`] is a pure function of them — the worker never touches
/// the live [`Training`] state, which keeps advancing the game while the grade is computed.
struct EvalJob {
    /// Monotonic id, for ordering the feedback panel and dropping grades from a since-left ruleset/deck.
    seq: u64,
    up: Card,
    hand: CardCol,
    rules: Ruleset,
    chosen: Move,
    /// The moves legal on the live hand (computed on the UI thread), so the basic reference is the best
    /// of *these* by the chart EVs rather than the unconditional two-card headline.
    allowed: Vec<Move>,
    /// The count-index deviation, already resolved on the UI thread (a cheap disk-cache lookup).
    indexed: Option<Move>,
    /// The round-start shoe for the finite live solve; `None` on the infinite deck (whose optimal play is
    /// the composition-independent basic play, taken from `basic_summary`).
    finite_shoe: Option<CardCol>,
    /// The memoized infinite-deck basic summary for this up-card, if already solved; `None` asks the
    /// worker to solve it and echo it back (in [`EvalResult::basic_summary`]) for memoization.
    basic_summary: Option<ColumnSummary>,
}

/// A finished decision grade streamed back from an eval worker (see [`Training::drain_evals`]).
struct EvalResult {
    seq: u64,
    /// The up-card the basic summary keys on (for memoization).
    up: Card,
    /// The graded decision, or `None` if the hand was somehow absent from the solved tree (unreachable in
    /// practice). Sent regardless so the pending-grade counter is always settled.
    mark: Option<DecisionMark>,
    /// `Some` only when the worker had to solve the infinite-deck basic summary, so the main thread can
    /// memoize it (keyed by `up`).
    basic_summary: Option<ColumnSummary>,
}

/// Grade one captured decision (the pure worker body behind [`Training::spawn_eval`]). Solves the basic
/// summary if it wasn't memoized, then the exact-optimal per-move EVs — the finite live solve over the
/// depleted composition, or, on the infinite deck, the basic cell's own composition-independent EVs —
/// and assembles the [`DecisionMark`]. Runs off the UI thread.
fn run_eval(job: EvalJob) -> EvalResult {
    let cat = categorize(&job.hand);
    // Basic strategy: the count-independent infinite-deck play. Use the memoized summary if the caller
    // had one, else solve it here and hand it back for memoization.
    let (basic_summary, computed) = match job.basic_summary {
        Some(summary) => (summary, None),
        None => {
            let summary = solve_on(InfiniteDeck {}, job.up, &job.rules).summary;
            (summary.clone(), Some(summary))
        }
    };
    let basic_cell = basic_summary.get(&cat);

    // Exact-optimal EVs for this hand: the live, count-aware solve over the depleted finite composition;
    // on the infinite deck the basic move EVs already are exact-optimal (EVs are composition-independent).
    let move_evs = match &job.finite_shoe {
        Some(shoe) => finite_move_evs(*shoe, &job.hand, job.up, &job.rules),
        None => basic_cell.map(|c| c.move_evs.clone()),
    };
    let mark = move_evs.map(|move_evs| {
        let optimal = best_move(&move_evs);
        let ev_optimal = move_evs[&optimal];
        let ev_chosen = move_evs.get(&job.chosen).copied().unwrap_or(ev_optimal);
        DecisionMark {
            chosen: job.chosen,
            // Basic strategy judged among only the legal moves (so a multi-card hand isn't graded
            // against the two-card-only Surrender/Double headline); on a two-card hand this is exactly
            // the chart headline, since the legal set then spans the cell's full move list.
            basic: basic_cell
                .map(|c| best_allowed_move(&c.move_evs, &job.allowed))
                .unwrap_or(optimal),
            indexed: job.indexed,
            optimal,
            ev_chosen,
            ev_optimal,
        }
    });

    EvalResult {
        seq: job.seq,
        up: job.up,
        mark,
        basic_summary: computed,
    }
}

/// The exact per-move EV map for `hand` against `up` solved over the *live, depleted* finite shoe — the
/// count-aware optimal reference. `None` if the hand is somehow absent from the tree. The live solve is
/// the cheap (~split-free) half of [`crate::simulation::build_evs`] when the hand is not a pair, plus a
/// single pair's split otherwise, so it stays well under a chart solve.
fn finite_move_evs(
    solver_shoe: CardCol,
    hand: &CardCol,
    up: Card,
    rules: &Ruleset,
) -> Option<HashMap<Move, f64>> {
    // Only the active hand's own split (if it is a pair) is relevant; other pairs needn't be solved.
    let split_evs = match pair_rank(hand) {
        Some(rank) if rules.max_split_hands >= 2 => {
            let pair = CardCol::from_hand(&[rank, rank]);
            pair_split_evs_for(&[pair], up, rules, |_| solver_shoe)
        }
        _ => HashMap::new(),
    };
    let tree = build_evs_with_splits(solver_shoe, up, rules, &split_evs);
    Some(tree.get(hand)?.1.clone())
}

/// The KO running count a fresh shoe starts at: the system's initial count for a finite `n`-deck shoe,
/// and a meaningless `0` on the infinite deck (which has no count). Shared by the constructor and the
/// reshuffle/sync paths so the three stay in step.
fn initial_count(shoe: &TrainShoe) -> i16 {
    match shoe {
        TrainShoe::Finite { n_decks, .. } => Ko::starting_count(*n_decks),
        TrainShoe::Infinite => 0,
    }
}

/// The move an ascending `(move, lo, hi)` count-index run list recommends at running count `ext`. The
/// runs cover `[runs.first().lo, runs.last().hi]` contiguously with the ends stretched to the window
/// edges, so a count past the window clamps onto the nearest open-ended run. `None` only for an empty
/// list (a category with no ladder).
fn run_move(runs: &[(Move, i16, i16)], ext: i16) -> Option<Move> {
    let lo = runs.first()?.1;
    let hi = runs.last()?.2;
    let e = ext.clamp(lo, hi);
    runs.iter()
        .find(|&&(_, l, h)| l <= e && e <= h)
        .map(|&(mv, _, _)| mv)
}

/// Every player action, the universe [`Training::spawn_eval`] filters through [`Training::allowed_move`]
/// to find the moves legal on the live hand.
const ALL_MOVES: [Move; 5] = [
    Move::Hit,
    Move::Stand,
    Move::Double,
    Move::Split,
    Move::Surrender,
];

/// The best of `allowed` by the cell's composition-independent EVs — the basic-strategy reference
/// restricted to the actions actually legal on the live hand. Falls back to the unrestricted argmax only
/// if none of `allowed` carries an EV (unreachable: Stand is always legal and always present).
fn best_allowed_move(move_evs: &HashMap<Move, f64>, allowed: &[Move]) -> Move {
    allowed
        .iter()
        .filter_map(|&m| move_evs.get(&m).map(|&ev| (m, ev)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(m, _)| m)
        .unwrap_or_else(|| best_move(move_evs))
}

/// The full move name, shared with the strategy tab's vocabulary.
pub(super) fn move_name(mv: Move) -> &'static str {
    match mv {
        Move::Hit => "Hit",
        Move::Stand => "Stand",
        Move::Double => "Double",
        Move::Split => "Split",
        Move::Surrender => "Surrender",
    }
}
