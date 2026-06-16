pub(crate) mod card;
pub(crate) mod count;
pub(crate) mod countshoe;
pub(crate) mod dealer;
pub(crate) mod diskcache;
pub(crate) mod hand;
mod legacy;
pub(crate) mod reach;
pub(crate) mod rules;
pub(crate) mod shoe;
pub(crate) mod simulation;
mod split;
#[cfg(test)]
mod test_support;
mod tui;

fn main() -> std::io::Result<()> {
    tui::run()
}
