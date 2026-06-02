# bacc-rs

A fast and memory-efficient baccarat engine. Deals rounds from a
multi-deck shoe and maintains the five standard scoreboards as
arbitrary-precision integers.

## Core types

- `BaccaratShoe` - wraps a shuffled shoe with the burn ritual and
  cut-card exhaustion. Implements `Iterator<Item = BaccaratRound>` so
  rounds can be consumed with a plain `for` loop.
- `BaccaratRound` - the resolved outcome of one round. Exposes player
  and banker card slices and a compact binary encoding of the outcome via `encode`.
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
- `BaccaratHand` - the cards dealt to one side, with hand value
  calculation, pair detection, and third-card flag.
- `BaccaratScoreboard` - tracks all five scoreboards as `BigUint`
  shift-registers. Decoupled from `BaccaratShoe`; the caller drives
  updates via `BaccaratScoreboard::update` after each round.

## Scoreboards

All five scoreboards are stored as `BigUint` shift-registers:

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
  as a run-length-encoded shift-register.
```
+--------+--------+--------+--------+--------+
|rrrrrrrp|rrrrrrrp|rrrrrrrp|rrrrrrrp|rrrrrrrp|
+--------+--------+--------+--------+--------+
|<-col1->|<-col2->|<-col3->|   ...  |<-coln->|

# p = prediction (0 = chaos, 1 = trend)
# rrrrrrr = row count (0~127)
```

## Usage

```rust
use bacc::{BaccaratScoreboard, BaccaratShoe};

let mut sb = BaccaratScoreboard::new();
let shoe = BaccaratShoe::new(8, 3, 0.965); // 8 decks, 3 shuffle passes, 96.5% penetration
for round in shoe {
    sb.update(&round);
    println!("{}", sb.bead_plate());
    println!("{}", sb.big_road());
    for road in sb.derived_roads() {
        println!("{}", road);
    }
}
```

## License

LGPL-3.0-only
