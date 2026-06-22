# bacc-rs

A shoe-dealer layer for baccarat. Wraps `shoe-rs` with baccarat
dealing rules and exposes rounds via the `Iterator` trait. Round
types and scoreboard tracking are provided by `bacc-core-rs`.

## Core types

`bacc-rs` exports one type:

- `BaccShoe` - wraps a shuffled shoe with the burn ritual and
  cut-card exhaustion. Implements `Iterator<Item = BaccRound>` so
  rounds can be consumed with a plain `for` loop.

The following types are from `bacc-core-rs` and used alongside `BaccShoe`:

- `BaccRound` - the resolved outcome of one round. Exposes player
  and banker card slices, a compact hex encoding of the full card
  sequence via `encode`, and a human-readable TOML fragment via `describe`.
  The derived outcome can be obtained via `BaccRound::outcome`, which
  returns a `BaccOutcome` with its own `encode` encoding:
```
+--------+--------+--------+--------+
|-xxxxxxx|xxttSSss|jjjjiiii|xx33ppww|
+--------+--------+--------+--------+

# ww = outcome (01=player, 10=banker, 11=tie)
# pp = pair flag (00=none, 01=player, 10=banker, 11=both)
# 33 = third card flag (00=none, 01=player, 10=banker, 11=both)
# iiii = player hand value (0~9)
# jjjj = banker hand value (0~9)
# (reserved) ss = two cards suited flag (00=none, 01=player, 10=banker, 11=both)
# (reserved) SS = three cards suited flag (00=none, 01=player, 10=banker, 11=both)
# (reserved) tt = trips flag (00=none, 01=player, 10=banker, 11=both)
```
- `BaccHand` - the cards dealt to one side, with hand value
  calculation, pair detection, and third-card flag.
- `BaccScoreboard` - tracks all five scoreboards as compact byte
  sequences. Decoupled from `BaccShoe`; the caller drives updates
  via `BaccScoreboard::update` after each round.

## Scoreboards

All five scoreboards are stored as compact byte sequences:

- **Bead plate** - two bytes per round, newest at bits 0-15.
```
+--------+--------+--------+--------+-----+--------+--------+
|xxxxvvvv|xx33ppww|xxxxvvvv|xx33ppww| ... |xxxxvvvv|xx33ppww|
+--------+--------+--------+--------+-----+--------+--------+
|<---  col 1  --->|<---  col 2  --->| ... |<---  col n  --->|

# ww = outcome (01=player, 10=banker, 11=tie)
# pp = pair flag (00=none, 01=player, 10=banker, 11=both)
# 33 = third card flag (00=none, 01=player, 10=banker, 11=both)
# vvvv = winning hand value (0~9)
```
- **Big road** - variable-width column shift-register, newest column at
  the low end.
```
+--------+--------+-----+--------+--------+--------+-----+--------+--------+-----+--------+--------+--------+--------+--------+
|ttttvvvv|xx33ppww| ... |ttttvvvv|xx33ppww|-rrrrrrr| ... |ttttvvvv|xx33ppww| ... |ttttvvvv|xx33ppww|ttttvvvv|xx33ppww|-rrrrrrr|
+--------+--------+-----+--------+--------+--------+-----+--------+--------+-----+--------+--------+--------+--------+--------+
|<---   row 1  -->| ... |<---   row j  -->|        | ... |<--   row 1   -->| ... |<--  row i-1  -->|<---   row i  -->|
|<---                   col 1                   -->| ... |<--                            col n                             -->|

# rrrrrrr = row count (0~127)
# ww = outcome (01=player, 10=banker, 11=tie)
# pp = pair flag (00=none, 01=player, 10=banker, 11=both)
# 33 = third card flag (00=none, 01=player, 10=banker, 11=both)
# vvvv = winning hand value (0~9)
# tttt = tie count (0~15)
```

- **Derived roads** - Big Eye Boy, Small Road, and Cockroach Pig, each
  as a run-length-encoded byte sequence.
```
+--------+--------+--------+--------+--------+
|rrrrrrrp|rrrrrrrp|rrrrrrrp|rrrrrrrp|rrrrrrrp|
+--------+--------+--------+--------+--------+
|<-col1->|<-col2->|<-col3->|   ...  |<-coln->|

# p = prediction (0 = chaos, 1 = trend)
# rrrrrrr = run length (0~127)
```

## Usage

```rust
use bacc::BaccShoe;
use bacc_core::BaccScoreboard;
use shoe::{Card, Shoe, DECK};

let mut cards: Vec<Card> = (0..8).flat_map(|_| DECK).collect();
cards.push(Card::Cut);
let last = cards.len() - 1;
cards.swap(last, 14); // place cut card at ~96.5% penetration

let mut sb = BaccScoreboard::new();
let shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
for round in shoe {
    sb.update(&round);
    print!("{}", round.describe());
    println!("[scoreboard]");
    println!("encoded = \"{}\"", sb.encode());
}
```

## License

LGPL-3.0-only
