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

- **Bead plate** - one byte per round, newest at bits 0-7.
```
+--------+--------+--------+--------+
|vvvvppww|vvvvppww|vvvvppww|vvvvppww|
+--------+--------+--------+--------+
|<-col1->|<-col2->|  ...   |<-coln->|

# ww = outcome (01=player, 10=banker, 11=tie)
# pp = pair flag (00=none, 01=player, 10=banker, 11=both)
# vvvv = winning hand value (0~9)
```
- **Big road** - variable-width column shift-register, newest column at
  the low end.
```
...+--------+--------+--------+--------+--------+--------+--------+--------+
...|-ttttttt|vvvvppww|-rrrrrrr|-ttttttt|vvvvppww|-ttttttt|vvvvppww|-rrrrrrr|
...+--------+--------+--------+--------+--------+--------+--------+--------+
   |<---   row 1  -->|  ...   |<--  row n-1  -->|<---   row n  -->|
   |<---   col 1  -->|  ...   |<--            col n            -->|

# rrrrrrr = row count (0~127)
# ww = outcome (01=player, 10=banker, 11=tie)
# pp = pair flag (00=none, 01=player, 10=banker, 11=both)
# vvvv = winning hand value (0~9)
# ttttttt = tie count (0~127)
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
use bacc::{write_outcome, write_bead_plate, write_big_road, write_derived_roads};

let mut sb = BaccaratScoreboard::new();
let shoe = BaccaratShoe::new(8, 3, 0.965); // 8 decks, 3 shuffle passes, 96.5% penetration
for round in shoe {
    sb.update(&round);
    write_outcome(&round, &mut std::io::stdout()).unwrap();
    write_bead_plate(sb.bead_plate(), &mut std::io::stdout()).unwrap();
    write_big_road(sb.big_road(), &mut std::io::stdout()).unwrap();
    write_derived_roads(sb.derived_roads(), &mut std::io::stdout()).unwrap();
}
```

Free functions for writing each scoreboard to any `std::io::Write` sink:
`write_outcome`, `write_bead_plate`, `write_big_road`,
`write_derived_roads`.

## License

LGPL-3.0-only
