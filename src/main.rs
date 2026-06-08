pub(crate) mod card;
pub(crate) mod dealer;
pub(crate) mod hand;
mod legacy;
pub(crate) mod rules;
pub(crate) mod shoe;
pub(crate) mod simulation;
mod split;
#[cfg(test)]
mod test_support;

use card::*;
use dealer::*;
use rules::*;
use shoe::*;
use simulation::*;

fn main() {
    println!("Hello, world!");
    println!("{}, {}, {}", Card::Pip(5), Card::Ten, Card::Ace);
    assert!(Card::Pip(2) < Card::Pip(3));
    assert!(Card::Pip(6) < Card::Ten);
    assert!(Card::Pip(9) > Card::Ace);
    assert!(Card::Ten > Card::Ace);

    let dd = CardCol::from_decks(4);
    println!("{} - {} total", dd, dd.len());

    let target_total = 16;
    let parts: Vec<_> = dd.weighted_partitions(target_total).collect();
    // println!("{:#?}", &parts);
    for (weight, hand) in parts.iter() {
        assert!(
            hand.iter()
                .map(|(c, n)| n as usize * c.hard() as usize)
                .sum::<usize>()
                == target_total.into()
        );
        println!("{}:\t{}", weight, hand);
    }
    println!("{} total partitions", parts.len());

    let rules = Ruleset::default();
    let dd = CardCol::from_decks(2);
    // `dealer_outcome_probs` is now always the raw (natural-included) distribution; apply
    // `remove_nat21` to show the peek-conditioned version side by side.
    let base_deal_probs = dealer_outcome_probs(CardCol::new(), &dd, rules.hs17);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", &base_deal_probs, norm);
    println!("{:?}\nnorm: {}", remove_nat21(base_deal_probs), norm);

    let dd = InfiniteDeck {};
    let base_deal_probs = dealer_outcome_probs(CardCol::new(), &dd, rules.hs17);
    let norm = base_deal_probs.values().sum::<f64>();
    assert!((norm - 1.0).abs() < 1e-12);
    println!("{:?}\nnorm: {}", base_deal_probs, norm);

    // NOTE: See https://wizardofodds.com/games/blackjack/appendix/9/1dh17r4/ for precise
    // comparisons
    let dd = CardCol::from_decks(2);
    // let dd = CardCol::half_deck();
    // let ev_map = build_evs(dd, Card::Pip(5), &rules);
    // let ev_map = build_evs(dd, Card::Pip(5), &rules);
    // let ev_map = build_evs(dd, Card::Pip(9), &rules);
    let ev_map = build_evs(dd, Card::Pip(5), &rules);
    let test_hand = CardCol::try_from("9A").unwrap();
    let soft20 = &ev_map[&test_hand];
    dbg!(soft20);

    let summary = summarize_evs(&ev_map);
    let strat = best_strategy(&summary);
    let mut sorted_strat: Vec<_> = strat.into_iter().collect();
    sorted_strat.sort_by_key(|(cat, _m)| *cat);
    for (cat, strat) in sorted_strat.into_iter() {
        println!("{}: {}", cat, strat);
    }
}
