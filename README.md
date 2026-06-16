# blackjack

A blackjack engine with a TUI to display basic strategy and index plays for a
rule set, along with a trainer for practice, drilling, and performance
evaluation.

Computations are typically exact up to some good approximations verified to
several decimal places (which are typically related to deep split arms).

## Usage

Build and launch this Rust project in release mode.

```
cargo run --release
```

Press `r` to change the ruleset and the basic strategy table will be populated
as the computations complete. Index plays are computed in the background and
will show up as indicators in the table when completed. Use `hjkl` to navigate
the chart, `<tab>` and `<s-tab>` to jump between panes, and press `<return>`
for a detailed view that will display any composition- or count-dependent
results.

To impose a count constraint on the strategy table (for instance, KO running
count >= -2, or HiLo true count > +1), press `c` to bring up the count
selection menu. This will modify the strategy table and update general
calculations like edge and insurance EV.

Computations are cached for fast retrieval on successive uses.

Training mode for the selected ruleset can be initiated by pressing `2`, and
the user can test their knowledge of basic strategy and counting. The
performance is monitored relative to the expectation values following
simplified basic strategy, specific basic strategy for the rules (as shown in
the table accessible by `1`), a count-specific variations strategy, and an
optimal strategy that would require perfect memory.
