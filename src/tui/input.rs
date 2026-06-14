//! Keyboard input: the per-mode key dispatch and the rules/count modal field editors. Extends
//! `impl App` (defined in [`super::app`]); [`handle_key`](App::handle_key) is the entry point the
//! [`event_loop`](App::event_loop) feeds.

use ratatui::crossterm::event::KeyCode;

use crate::count::CountCmp;
use crate::hand::Move;
use crate::rules::{BjPayout, PeekRule, PeekSurrender};

use super::app::{App, Mode};
use super::config::{DECK_OPTIONS, SPLIT_OPTIONS};
use super::training::Phase;
use super::{PANES, Tab, UP_CARDS};

/// Number of editable fields in the rules modal.
const RULES_FIELDS: usize = 8;

/// Number of fields in the count modal (enabled, comparison, value).
const COUNT_FIELDS: usize = 3;

impl App {
    /// Handle one key press. Returns `true` to quit.
    pub(super) fn handle_key(&mut self, code: KeyCode) -> bool {
        // The rules editor is a shared overlay: handle it before any tab-specific routing so every tab
        // (the strategy chart, the trainer, and any future tab) opens and edits the same modal.
        if self.mode == Mode::Rules {
            self.handle_rules(code);
            return false;
        }
        // Otherwise the training tab runs its own modeless key handling (it has no chart-only modals).
        if self.tab == Tab::Training {
            return self.handle_training(code);
        }
        match self.mode {
            Mode::Normal => return self.handle_normal(code),
            Mode::Popup => self.handle_popup(code),
            Mode::Count => self.handle_count(code),
            Mode::Rules => {} // handled above, before tab routing
        }
        false
    }

    /// Open the shared rules-editing modal, snapshotting the current `(rules, shoe)` so a change can be
    /// detected on close. Shared by the strategy tab (`r`) and the training tab (`r` outside the
    /// player's turn).
    fn open_rules(&mut self) {
        self.rules_snapshot = (self.rules, self.shoe);
        self.rules_sel = 0;
        self.mode = Mode::Rules;
    }

    fn handle_normal(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('q') => return true,
            KeyCode::Char('2') => self.set_tab(Tab::Training),
            KeyCode::Enter | KeyCode::Char(' ') => self.mode = Mode::Popup,
            KeyCode::Char('r') => self.open_rules(),
            KeyCode::Char('c') => {
                self.count_snapshot = (self.count_on, self.count);
                self.count_sel = 0;
                self.mode = Mode::Count;
            }
            _ => self.move_cursor(code),
        }
        false
    }

    /// In the popup, motion keys still move the selection (the popup tracks it live); Esc closes.
    fn handle_popup(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => self.mode = Mode::Normal,
            _ => self.move_cursor(code),
        }
    }

    fn move_cursor(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('h') | KeyCode::Left => {
                self.cursor.col = self.cursor.col.saturating_sub(1)
            }
            KeyCode::Char('l') | KeyCode::Right => {
                self.cursor.col = (self.cursor.col + 1).min(UP_CARDS.len() - 1)
            }
            KeyCode::Char('k') | KeyCode::Up => self.cursor.row = self.cursor.row.saturating_sub(1),
            KeyCode::Char('j') | KeyCode::Down => {
                let len = self.active_pane().rows().len();
                self.cursor.row = (self.cursor.row + 1).min(len - 1);
            }
            KeyCode::Tab => {
                self.cursor.pane = (self.cursor.pane + 1) % PANES.len();
                self.clamp_row();
            }
            KeyCode::BackTab => {
                self.cursor.pane = (self.cursor.pane + PANES.len() - 1) % PANES.len();
                self.clamp_row();
            }
            _ => {}
        }
    }

    fn handle_rules(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('r') | KeyCode::Char('q') => {
                if (self.rules, self.shoe) != self.rules_snapshot {
                    self.recompute();
                    // Re-init the trainer's live shoe for the new rules/deck (it shares the same modal).
                    self.training.on_rules_changed(self.shoe);
                }
                self.mode = Mode::Normal;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.rules_sel = (self.rules_sel + 1) % RULES_FIELDS;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.rules_sel = (self.rules_sel + RULES_FIELDS - 1) % RULES_FIELDS;
            }
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Char(' ') => self.edit_rule(1),
            KeyCode::Char('h') | KeyCode::Left => self.edit_rule(-1),
            _ => {}
        }
    }

    /// Change the currently selected rules field by `delta` (booleans toggle regardless of sign).
    fn edit_rule(&mut self, delta: i32) {
        match self.rules_sel {
            0 => {
                let i = DECK_OPTIONS
                    .iter()
                    .position(|&d| d == self.shoe)
                    .unwrap_or(0) as i32;
                let n = DECK_OPTIONS.len() as i32;
                self.shoe = DECK_OPTIONS[(i + delta).rem_euclid(n) as usize];
            }
            1 => self.rules.hs17 = !self.rules.hs17,
            2 => self.rules.das = !self.rules.das,
            3 => {
                // Toggle the peek, carrying the surrender choice across as far as the target state
                // allows: a no-peek game cannot hold *late* surrender, so it lands on early instead.
                self.rules.peek = match self.rules.peek {
                    PeekRule::Peek(PeekSurrender::None) => PeekRule::NoPeek {
                        early_surrender: false,
                    },
                    PeekRule::Peek(_) => PeekRule::NoPeek {
                        early_surrender: true,
                    },
                    PeekRule::NoPeek {
                        early_surrender: false,
                    } => PeekRule::Peek(PeekSurrender::None),
                    PeekRule::NoPeek {
                        early_surrender: true,
                    } => PeekRule::Peek(PeekSurrender::Early),
                };
            }
            4 => {
                self.rules.bj_payout = match self.rules.bj_payout {
                    BjPayout::ThreeToTwo => BjPayout::SixToFive,
                    BjPayout::SixToFive => BjPayout::ThreeToTwo,
                };
            }
            5 => {
                self.rules.peek = match self.rules.peek {
                    PeekRule::Peek(s) => {
                        let order = [
                            PeekSurrender::None,
                            PeekSurrender::Early,
                            PeekSurrender::Late,
                        ];
                        let i = order.iter().position(|&x| x == s).unwrap_or(0) as i32;
                        PeekRule::Peek(order[(i + delta).rem_euclid(3) as usize])
                    }
                    // No peek: late surrender is impossible, so this is just an early-surrender toggle.
                    PeekRule::NoPeek { early_surrender } => PeekRule::NoPeek {
                        early_surrender: !early_surrender,
                    },
                };
            }
            6 => {
                let v = self.rules.max_split_hands as i32 + delta;
                self.rules.max_split_hands = v.clamp(1, 4) as u8;
            }
            7 => {
                let i = SPLIT_OPTIONS
                    .iter()
                    .position(|&s| s == self.rules.split_cards)
                    .unwrap_or(2) as i32;
                let n = SPLIT_OPTIONS.len() as i32;
                self.rules.split_cards = SPLIT_OPTIONS[(i + delta).rem_euclid(n) as usize];
            }
            _ => {}
        }
    }

    fn handle_count(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('c') | KeyCode::Char('q') => {
                if (self.count_on, self.count) != self.count_snapshot {
                    self.recompute();
                }
                self.mode = Mode::Normal;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.count_sel = (self.count_sel + 1) % COUNT_FIELDS;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.count_sel = (self.count_sel + COUNT_FIELDS - 1) % COUNT_FIELDS;
            }
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Char(' ') => self.edit_count(1),
            KeyCode::Char('h') | KeyCode::Left => self.edit_count(-1),
            _ => {}
        }
    }

    /// Key handling for the training tab. Modeless apart from the count-quiz overlay; returns `true` to
    /// quit. The game-advancing keys route into the (stubbed) [`Training`](super::training::Training)
    /// simulation seam.
    fn handle_training(&mut self, code: KeyCode) -> bool {
        // The count-quiz overlay captures input while open: adjust the guess and submit.
        if self.training.entering_count {
            match code {
                KeyCode::Char('h') | KeyCode::Left => self.training.count_entry -= 1,
                KeyCode::Char('l') | KeyCode::Right => self.training.count_entry += 1,
                KeyCode::Enter | KeyCode::Char(' ') => self.training.submit_count(),
                KeyCode::Esc | KeyCode::Char('n') => self.training.entering_count = false,
                _ => {}
            }
            return false;
        }

        match code {
            KeyCode::Char('q') => return true,
            KeyCode::Char('1') => self.set_tab(Tab::Strategy),
            // Open the running-count quiz — finite shoe only (the infinite deck has no count).
            KeyCode::Char('n') if self.training.is_finite() => {
                self.training.entering_count = true;
            }
            // Deal a fresh round from the Ready or Settled phase (Enter or `d`).
            KeyCode::Enter | KeyCode::Char('d')
                if matches!(self.training.phase, Phase::Ready | Phase::Settled) =>
            {
                let rules = self.rules;
                self.training.deal(&rules);
            }
            // `r` opens the shared rules editor — except during the player's turn, where it is Surrender
            // (handled by the player-action arm below). Editing rules mid-round would abandon it anyway.
            KeyCode::Char('r') if self.training.phase != Phase::Player => self.open_rules(),
            // Player actions, only while it is the player's turn.
            _ if self.training.phase == Phase::Player => {
                if let Some(mv) = training_move(code) {
                    let rules = self.rules;
                    self.training.player_move(mv, &rules);
                }
            }
            _ => {}
        }
        false
    }

    /// Change the selected count-modal field by `delta`: toggle enabled, cycle the comparison, or
    /// step the running-count value.
    fn edit_count(&mut self, delta: i32) {
        match self.count_sel {
            0 => self.count_on = !self.count_on,
            1 => {
                let order = [CountCmp::Le, CountCmp::Eq, CountCmp::Ge];
                let i = order.iter().position(|&c| c == self.count.cmp).unwrap_or(1) as i32;
                self.count.cmp = order[(i + delta).rem_euclid(order.len() as i32) as usize];
            }
            2 => self.count.external = (self.count.external as i32 + delta).clamp(-60, 60) as i16,
            _ => {}
        }
    }
}

/// Map a training-tab key to a player [`Move`], or `None` if it isn't an action key. The double key
/// (`d`) only reaches here in the player phase — the deal handler claims it in Ready/Settled.
fn training_move(code: KeyCode) -> Option<Move> {
    match code {
        KeyCode::Char('h') => Some(Move::Hit),
        KeyCode::Char('s') => Some(Move::Stand),
        KeyCode::Char('d') => Some(Move::Double),
        KeyCode::Char('p') => Some(Move::Split),
        KeyCode::Char('r') => Some(Move::Surrender),
        _ => None,
    }
}
