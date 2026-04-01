//! Baccarat engine library.
//!
//! `bacc` deals baccarat rounds from a multi-deck shoe and maintains the five
//! standard scoreboards as arbitrary-precision integers.
//!
//! # Core types
//!
//! - [`BaccaratShoe`] - wraps `shoe::Shoe` with baccarat dealing rules. Implements
//!   [`Iterator`]`<Item = `[`BaccaratRound`]`>` so rounds can be consumed with
//!   a plain `for` loop.
//! - [`BaccaratScoreboard`] - tracks the five scoreboards.
//! - [`BaccaratRound`] - the resolved outcome of one round. Exposes the player
//!   and banker card slices and a compact [`BaccaratRound::onehot`] encoding of
//!   the outcome.
//! - [`BaccaratHand`] - the cards dealt to one side, with hand value calculation,
//!   pair detection, and third-card flag.
//!
//! # Scoreboards
//!
//! All five scoreboards are stored as [`BigUint`] shift-registers. Call
//! [`BaccaratScoreboard::update`] after each round to advance them:
//!
//! - **Bead plate** ([`BaccaratScoreboard::bead_plate`]) - one byte per round,
//!   newest at bits 0-7. Each byte encodes the winner's hand value, pair flags,
//!   and outcome.
//! - **Big road** ([`BaccaratScoreboard::big_road`]) - variable-width column
//!   shift-register. Each column occupies `2n + 1` bytes where `n` is the row
//!   count; newest column at the low end.
//! - **Derived roads** ([`BaccaratScoreboard::derived_roads`]) - Big Eye Boy,
//!   Small Road, and Cockroach Pig, one run-length-encoded shift-register each.
//!
//! # Usage
//!
//! ```rust
//! use bacc::{BaccaratScoreboard, BaccaratShoe};
//! use bacc::{write_outcome, write_bead_plate, write_big_road, write_derived_roads};
//! use num_bigint::BigUint;
//!
//! let mut sb = BaccaratScoreboard::new();
//! let shoe = BaccaratShoe::new(8, 3, 0.965); // 8 deck, 3 shuffle passes, 96.5% penetration
//! for round in shoe {
//!     sb.update(&round);
//!     let bead_plate: &BigUint = sb.bead_plate();
//!     let big_road: &BigUint = sb.big_road();
//!     let derived_roads: &[BigUint; 3] = sb.derived_roads();
//!     write_outcome(&round, &mut std::io::stdout()).unwrap();
//!     write_bead_plate(bead_plate, &mut std::io::stdout()).unwrap();
//!     write_big_road(big_road, &mut std::io::stdout()).unwrap();
//!     write_derived_roads(derived_roads, &mut std::io::stdout()).unwrap();
//! }
//! ```

use kev::{CardInt, Rank};
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use num_traits::{One, Zero};
use shoe::{Card, Shoe};
use std::collections::VecDeque;

/// Returns the pip value of a single [`CardInt`].
///
/// | Rank                   | Value |
/// |------------------------|-------|
/// | Ace                    | 1     |
/// | 2-9                    | pip   |
/// | Ten, Jack, Queen, King | 10    |
fn pip_value(card: CardInt) -> u8 {
    match card.rank() {
        Rank::Ace => 1,
        Rank::King | Rank::Queen | Rank::Jack | Rank::Ten => 10,
        Rank::Nine => 9,
        Rank::Eight => 8,
        Rank::Seven => 7,
        Rank::Six => 6,
        Rank::Five => 5,
        Rank::Four => 4,
        Rank::Trey => 3,
        Rank::Deuce => 2,
    }
}

/// Returns `true` if neither hand holds a natural (8 or 9).
///
/// A natural ends the round immediately; third-card rules only apply when both
/// sides score 0-7 on their initial two cards.
///
/// # Panics
///
/// Panics if either hand already holds a third card.
fn no_natural(p: &BaccaratHand, b: &BaccaratHand) -> bool {
    assert!(!p.has_third() && !b.has_third());
    matches!((p.value(), b.value()), (0..=7, 0..=7))
}

/// Returns `true` if `hand` scores 6 or 7 and must stand pat.
///
/// Used for both player and banker. A hand valued 6 or 7 stands without
/// drawing a third card.
///
/// # Panics
///
/// Panics if `hand` already holds a third card.
fn stand_pat(hand: &BaccaratHand) -> bool {
    assert!(!hand.has_third());
    matches!(hand.value(), 6 | 7)
}

/// Returns `true` if the banker draws a third card given the player's third card.
///
/// Applies the standard banker drawing table. Called only when the player has
/// already drawn a third card and neither side holds a natural.
///
/// | Banker score | Draws when player's third card pip is |
/// |--------------|---------------------------------------|
/// | 0-2          | Always draws                          |
/// | 3            | Any except 8                          |
/// | 4            | 2-7                                   |
/// | 5            | 4-7                                   |
/// | 6            | 6-7                                   |
/// | 7            | Never draws                           |
///
/// # Panics
///
/// Panics if `banker_hand` already holds a third card.
fn banker_take_third(banker_hand: &BaccaratHand, player_third_card: CardInt) -> bool {
    assert!(!banker_hand.has_third());
    let p = pip_value(player_third_card);
    match banker_hand.value() {
        0..=2 => true,
        3 => p != 8,
        4 => matches!(p, 2..=7),
        5 => matches!(p, 4..=7),
        6 => matches!(p, 6 | 7),
        _ => false,
    }
}

/// Writes a [`BaccaratRound`] summary to `out`.
///
/// # Errors
///
/// Returns any I/O error produced by `out`.
pub fn write_outcome<W: std::io::Write>(input: &BaccaratRound, out: &mut W) -> std::io::Result<()> {
    let onehot = input.onehot();
    let outcome = match onehot & 0x3 {
        1 => "Player",
        2 => "Banker",
        _ => "Tie",
    };
    let player_value = onehot >> 8 & 0xF;
    let banker_value = onehot >> 12 & 0xF;
    writeln!(
        out,
        "Outcome: {outcome} (player={player_value} banker={banker_value})"
    )
}

/// Prints the bead plate scoreboard to `out`.
///
/// # Errors
///
/// Returns any I/O error produced by `out`.
///
/// # Panics
///
/// Panics if the internal `bead_plate` encoding is corrupt (column count or
/// marker fields exceed their expected bit widths).
pub fn write_bead_plate<W: std::io::Write>(input: &BigUint, out: &mut W) -> std::io::Result<()> {
    let mut bead_plate = input.clone();
    let mut deque: VecDeque<String> = VecDeque::new();
    loop {
        let bead = (&bead_plate & BigUint::from(0xFFu8)).to_u8().unwrap();
        if bead == 0 {
            break;
        }
        let mut marker = String::new();
        if bead & 8 > 0 {
            marker.push('.');
        }
        match bead & 3 {
            1 => marker.push('P'),
            2 => marker.push('B'),
            3 => marker.push('T'),
            _ => unreachable!("error in encode_bead()"),
        }
        let value = (u32::from(bead) & 0xF0) >> 4;
        marker.push(char::from_digit(value, 10).expect("digit must be in the range 0-9"));
        if (bead & 4) > 0 {
            marker.push('.');
        }
        deque.push_front(marker);
        bead_plate >>= 8;
    }
    writeln!(out, "Bead Plate:")?;
    for i in &deque {
        write!(out, "{i} ")?;
    }
    writeln!(out)
}

/// Prints the big road scoreboard to `out`.
///
/// # Errors
///
/// Returns any I/O error produced by `out`.
///
/// # Panics
///
/// Panics if the internal `big_road` encoding is corrupt (column count or
/// marker fields exceed their expected bit widths).
pub fn write_big_road<W: std::io::Write>(input: &BigUint, out: &mut W) -> std::io::Result<()> {
    let mut big_road = input.clone();
    let mut deque: VecDeque<[u8; 2]> = VecDeque::new();
    loop {
        let count = (&big_road & BigUint::from(0xFFu8)).to_u8().unwrap();
        if count == 0 {
            break;
        }
        let marker = ((&big_road & BigUint::from(0x300u16)) >> 8u8)
            .to_u8()
            .unwrap();
        deque.push_front([marker, count]);
        big_road >>= 8 * (2 * count + 1) as usize;
    }
    if deque.is_empty() {
        Ok(())
    } else {
        writeln!(out, "Big Road:")?;
        for i in &deque {
            let marker = match i[0] {
                1 => "P",
                2 => "B",
                _ => unreachable!("error in update_big_road()"),
            };
            write!(out, "|{marker}")?;
        }
        writeln!(out, "|")?;
        for i in &deque {
            write!(out, "|{}", i[1])?;
        }
        writeln!(out, "|")
    }
}

/// Prints the derived road scoreboards to `out`.
///
/// # Errors
///
/// Returns any I/O error produced by `out`.
///
/// # Panics
///
/// Panics if the internal `derived_roads` encoding is corrupt (column count or
/// marker fields exceed their expected bit widths).
pub fn write_derived_roads<W: std::io::Write>(
    input: &[BigUint; 3],
    out: &mut W,
) -> std::io::Result<()> {
    let labels = ["Big Eye Boy:", "Small Road:", "Cockroach Pig:"];
    for (number, label) in input.iter().zip(labels.iter()) {
        let mut derived_road = number.clone();
        let mut deque: VecDeque<[u8; 2]> = VecDeque::new();
        loop {
            let count = (&derived_road & BigUint::from(0xFEu8)).to_u8().unwrap() >> 1;
            if count == 0 {
                break;
            }
            let marker = (&derived_road & BigUint::one()).to_u8().unwrap();
            deque.push_front([marker, count]);
            derived_road >>= 8;
        }
        if deque.is_empty() {
            continue;
        }
        writeln!(out, "{label}")?;
        for i in &deque {
            let marker = match i[0] {
                0 => "C",
                1 => "T",
                _ => unreachable!("error in update_derived_roads()"),
            };
            write!(out, "|{marker}")?;
        }
        writeln!(out, "|")?;
        for i in &deque {
            write!(out, "|{}", i[1])?;
        }
        writeln!(out, "|")?;
    }
    Ok(())
}

/// A baccarat hand holding the cards dealt to one side (player or banker).
#[derive(Default)]
pub struct BaccaratHand {
    cards: Vec<CardInt>,
}

impl BaccaratHand {
    /// Adds `card` to this hand.
    pub fn take(&mut self, card: &CardInt) {
        self.cards.push(*card);
    }

    /// Returns the baccarat point value of the hand (0-9).
    ///
    /// Sums the pip value for each card in the hand and reduces the total
    /// modulo 10 - matching standard baccarat scoring rules.
    #[must_use]
    pub fn value(&self) -> u8 {
        let total: u8 = self.cards.iter().map(|&x| pip_value(x)).sum();
        total % 10
    }

    /// Returns `true` if the first two cards share the same rank.
    ///
    /// # Panics
    ///
    /// Panics if the hand contains fewer than two cards.
    #[must_use]
    pub fn is_pair(&self) -> bool {
        self.cards[0].rank() == self.cards[1].rank()
    }

    /// Returns `true` if the hand contains exactly three cards.
    #[must_use]
    pub fn has_third(&self) -> bool {
        self.cards.len() == 3
    }

    /// Returns a slice of the cards held in this hand.
    #[must_use]
    pub fn cards(&self) -> &[CardInt] {
        self.cards.as_slice()
    }
}

/// A single resolved baccarat round, holding the final hands for both sides.
pub struct BaccaratRound {
    player: BaccaratHand,
    banker: BaccaratHand,
}

impl BaccaratRound {
    /// Encodes the outcome and key facts of this round into a single `u32`.
    ///
    /// ## Bit layout
    ///
    /// | Bits  | Field             | Values                                               |
    /// |-------|-------------------|------------------------------------------------------|
    /// | 0-1   | Outcome           | `1` = player, `2` = banker, `3` = tie                |
    /// | 2     | Player pair       | `1` if player's first two cards share a rank         |
    /// | 3     | Banker pair       | `1` if banker's first two cards share a rank         |
    /// | 4     | Player third card | `1` if player drew a third card                      |
    /// | 5     | Banker third card | `1` if banker drew a third card                      |
    /// | 8-11  | Player hand value | Player's hand value 0-9 (sum of card values mod 10)  |
    /// | 12-15 | Banker hand value | Banker's hand value 0-9 (sum of card values mod 10)  |
    ///
    /// # Panics
    ///
    /// Panics if either hand contains fewer than two cards, as
    /// [`BaccaratHand::is_pair`] indexes `cards[0]` and `cards[1]` unconditionally.
    #[must_use]
    pub fn onehot(&self) -> u32 {
        let mut onehot: u32 = 3;
        let player_hand_value: u8 = self.player.value();
        let banker_hand_value: u8 = self.banker.value();
        if player_hand_value > banker_hand_value {
            onehot &= 1;
        } else if player_hand_value < banker_hand_value {
            onehot &= 2;
        }
        onehot |= u32::from(self.player.is_pair()) << 2 | u32::from(self.banker.is_pair()) << 3;
        onehot |= u32::from(self.player.has_third()) << 4 | u32::from(self.banker.has_third()) << 5;
        onehot |= u32::from(player_hand_value) << 8 | u32::from(banker_hand_value) << 12;
        onehot
    }

    /// Returns a slice of the player's cards.
    #[must_use]
    pub fn player_cards(&self) -> &[CardInt] {
        self.player.cards()
    }

    /// Returns a slice of the banker's cards.
    #[must_use]
    pub fn banker_cards(&self) -> &[CardInt] {
        self.banker.cards()
    }
}

/// Tracks the five standard baccarat scoreboards for a running shoe.
///
/// Call [`update`] after each round dealt by [`BaccaratShoe`] to advance them.
///
/// [`update`]: BaccaratScoreboard::update
#[derive(Default)]
pub struct BaccaratScoreboard {
    // Shift-register of bead bytes, newest at bits 0-7.
    // Each byte: bits 7-4 = winner's hand value, bits 3-2 = pair flags, bits 1-0 = outcome.
    bead_plate: BigUint,
    // Variable-width column shift-register. Each column occupies (1 + 2n) bytes (n = row count),
    // packed at the low end:
    //   byte 0       - row_count n
    //   bytes 1-2    - bead + tie_count of the most recent row
    //   bytes 3-4    - bead + tie_count of the row before that
    //   ...          - earlier rows; previous columns are packed above
    big_road: BigUint,
    // [Big Eye Boy, Small Road, Cockroach Pig] - one run-length-encoded shift-register each.
    // Each byte: bits 7-1 = run length, bit 0 = icon (1 = red, 0 = blue).
    // Matching icon: byte incremented by 2 (run length in bits 7-1 incremented by 1). New icon: fresh byte pushed onto the low end.
    derived_roads: [BigUint; 3],
}

impl BaccaratScoreboard {
    /// Creates a new [`BaccaratScoreboard`] with all scoreboards zeroed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates all five scoreboards immediately after a completed round.
    pub fn update(&mut self, round: &BaccaratRound) {
        let bead = Self::bead_byte(round.onehot());
        let is_tie = bead & 0x3 == 0x3;
        self.update_bead_plate(bead);
        self.update_big_road(bead, is_tie);
        if !is_tie {
            self.update_derived_roads();
        }
    }

    /// Resets all five scoreboards to zero.
    pub fn clear(&mut self) {
        self.bead_plate = BigUint::ZERO;
        self.big_road = BigUint::ZERO;
        self.derived_roads = [BigUint::ZERO, BigUint::ZERO, BigUint::ZERO];
    }

    /// Returns the bead plate as a shift-register of bead bytes, newest at bits 0-7.
    #[must_use]
    pub fn bead_plate(&self) -> &BigUint {
        &self.bead_plate
    }

    /// Returns the big road as a variable-width column shift-register, newest column at the low end.
    #[must_use]
    pub fn big_road(&self) -> &BigUint {
        &self.big_road
    }

    /// Returns the three derived roads - Big Eye Boy, Small Road, and Cockroach Pig - as
    /// run-length-encoded shift-registers, one per element.
    #[must_use]
    pub fn derived_roads(&self) -> &[BigUint; 3] {
        &self.derived_roads
    }

    /// Encodes a round's `onehot` value into a single bead byte.
    ///
    /// | Bits | Content |
    /// |------|---------|
    /// | 7-4  | Winner's hand value (0-9) |
    /// | 3    | Banker pair flag |
    /// | 2    | Player pair flag |
    /// | 1-0  | Outcome (`1` = player, `2` = banker, `3` = tie) |
    ///
    /// Banker wins use the banker's hand value in the high nibble (bits 12-15 of `onehot`);
    /// player wins and ties use the player's hand value (bits 8-11 of `onehot`).
    fn bead_byte(onehot: u32) -> u8 {
        let low_nib: u8 = (onehot & 0xF) as u8;
        let n: u8 = match low_nib & 0x3 {
            2 => 8,
            _ => 4,
        };
        low_nib | (onehot >> n & 0xF0) as u8
    }

    /// Prepends `bead` to the bead-plate shift-register; most recent round is at bits 0-7.
    fn update_bead_plate(&mut self, bead: u8) {
        self.bead_plate <<= 8;
        self.bead_plate |= BigUint::from(bead);
    }

    /// Advances the big road by one round.
    ///
    /// Five transitions driven by `big_road` state and `last_outcome` (bits 8-9 of `big_road`):
    /// - **Initial round** (`big_road` is zero) - write first bead and set `row_count` to 1.
    /// - **Tie** - increment the tie counter at bits 16-23; column unchanged.
    /// - **Opening ties resolved** - preserve tie count and add 1 (recovering the count missed
    ///   by the initial-round path), write first real bead, reset `row_count` to 1.
    /// - **Column hit** (same side wins) - grow current column by one row.
    /// - **New column** (opposite side wins) - archive current column, start a fresh one.
    fn update_big_road(&mut self, bead: u8, is_tie: bool) {
        let bead_shl: BigUint = BigUint::from(bead) << 8;
        if self.big_road.is_zero() {
            if is_tie {
                self.big_road = bead_shl + BigUint::from(0x1_0000_u32);
            } else {
                self.big_road = bead_shl | BigUint::one();
            }
            return;
        }
        // Outcome bits of the current column's topmost row sit at bits 8-9.
        let last_outcome: u8 = ((&self.big_road & BigUint::from(0x300u16)) >> 8u8)
            .to_u8()
            .unwrap();
        let is_shoe_tie_start: bool = last_outcome == 0x3;
        let is_column_hit: bool = last_outcome == (bead & 0x3);
        if is_tie {
            self.big_road += BigUint::from(0x1_0000u32);
        } else if is_shoe_tie_start {
            // TODO: verify against the baccarat scoreboard spec that discarding the column
            // history here is correct for a shoe that opens with one or more consecutive ties.
            let tie_cnt = &self.big_road & BigUint::from(0xFF_0000_u32);
            self.big_road = tie_cnt | bead_shl | BigUint::one();
        } else if is_column_hit {
            // Remove stale row_count byte (>> 8), shift existing rows up 24 bits, insert new row.
            let row_cnt = (&self.big_road & BigUint::from(0xFFu8)) + BigUint::one();
            self.big_road >>= 8;
            self.big_road <<= 24;
            self.big_road |= bead_shl | row_cnt;
        } else {
            self.big_road <<= 24;
            self.big_road |= bead_shl | BigUint::one();
        }
    }

    /// Walks the variable-width big-road encoding and returns the row count of the five
    /// most recent columns (index 0 = current column, 1 = one column back, ...).
    ///
    /// A column of height `n` occupies `2n + 1` bytes, so the bit-skip is `8 * (2n + 1)`.
    fn col_heights(&self) -> [u8; 5] {
        let mut heights = [0u8; 5];
        let mut temp = self.big_road.clone();
        for slot in &mut heights {
            *slot = (&temp & BigUint::from(0xFFu8)).to_u8().unwrap();
            temp >>= 8 * (2 * usize::from(*slot) + 1);
        }
        heights
    }

    /// Pushes one run-length-encoded icon onto `derived_roads[road_idx]`.
    ///
    /// Each byte encodes `bits 7-1` = run length and `bit 0` = icon (`1` = red, `0` = blue).
    /// A matching icon increments the byte by 2 (run length in bits 7-1 incremented by 1);
    /// a new icon pushes a fresh byte onto the low end.
    fn push_derived_road_icon(&mut self, road_idx: usize, icon: u8) {
        let road = &mut self.derived_roads[road_idx];
        if road.is_zero() {
            *road = BigUint::from(2u8) | BigUint::from(icon);
        } else {
            let last_icon = (&*road & BigUint::one()).to_u8().unwrap();
            if icon == last_icon {
                *road += BigUint::from(2u8);
            } else {
                *road <<= 8;
                *road |= BigUint::from(2u8) | BigUint::from(icon);
            }
        }
    }

    /// Updates the three derived roads (Big Eye Boy, Small Road, Cockroach Pig).
    ///
    /// Called only for non-tie rounds. For each road at offset `i` (1-3), the icon is determined
    /// by comparing column heights: red (`1`) if the reference columns do not follow the expected
    /// pattern, blue (`0`) if they do. See [`push_derived_road_icon`] for encoding details.
    ///
    /// [`push_derived_road_icon`]: Self::push_derived_road_icon
    fn update_derived_roads(&mut self) {
        let col = self.col_heights();
        for i in 1..=3usize {
            // Need a reference column (i+1 back) or a growing current column with a comparison
            // column (i back) to produce a meaningful icon.
            let has_ref_col = col[i + 1] > 0;
            let has_growing_col = col[i] > 0 && col[0] > 1;
            if !(has_ref_col || has_growing_col) {
                continue;
            }
            // New column (height 1): red if adjacent reference columns are equal in height.
            // Growing column: red if current height does not trail column i by exactly one.
            let icon: u8 = if col[0] == 1 {
                u8::from(col[i] == col[i + 1])
            } else {
                u8::from(col[0] != col[i] + 1)
            };
            self.push_derived_road_icon(i - 1, icon);
        }
    }
}

/// A baccarat shoe that deals [`BaccaratRound`]s via the [`Iterator`] trait.
///
/// Wraps `shoe::Shoe` with baccarat-specific dealing: the burn ritual is
/// applied on construction (first card exposed, that many cards discarded),
/// and each round is dealt using the standard third-card rules.
pub struct BaccaratShoe {
    shoe: shoe::Shoe,
    // Set after the cut card is reached; the current hand plays out and then the iterator stops.
    is_exhausted: bool,
}

impl BaccaratShoe {
    /// Creates a new [`BaccaratShoe`] with a freshly shuffled shoe of `num_decks` decks.
    ///
    /// `passes` is the number of shuffle passes applied to the shoe before cutting.
    /// `pen` sets what fraction of the shoe is dealt before the cut card (0.5 = 50%, 0.9 = 90%).
    /// The burn ritual is then applied: the first card is exposed and that many cards are
    /// discarded before play begins.
    ///
    /// # Panics
    ///
    /// Panics if fewer than 12 cards would remain after the cut card for the given `num_decks`.
    #[must_use]
    pub fn new(num_decks: usize, passes: u8, pen: f32) -> Self {
        let mut shoe: Shoe = Shoe::new(num_decks);
        shoe.shuffle(passes);
        shoe.cut(pen);
        assert!(
            shoe.stub_size() >= 12,
            "pen leaves fewer than 12 cards after the cut card for a {num_decks}-deck shoe"
        );
        Self::from_shoe(shoe)
    }

    fn from_shoe(mut shoe: Shoe) -> Self {
        let first_card = match shoe.deal().expect("shoe is non-empty") {
            Card::Play(c) => c,
            Card::Cut => unreachable!("cut card dealt as first card"),
        };
        shoe.burn(pip_value(first_card) as usize);
        Self {
            shoe,
            is_exhausted: false,
        }
    }

    fn pull(&mut self) -> CardInt {
        loop {
            match self.shoe.deal().expect("shoe is non-empty") {
                Card::Play(c) => break c,
                Card::Cut => {}
            }
        }
    }
}

impl From<Vec<Card>> for BaccaratShoe {
    fn from(cards: Vec<Card>) -> Self {
        Self::from_shoe(Shoe::from(cards))
    }
}

impl Iterator for BaccaratShoe {
    type Item = BaccaratRound;

    /// Deals and returns the next [`BaccaratRound`], or `None` when the shoe
    /// is exhausted.
    ///
    /// Each call deals two cards to the player and two to the banker, then
    /// applies the standard third-card rules:
    ///
    /// - If either side holds a natural (8 or 9) no further cards are drawn.
    /// - Otherwise the player draws on 0-5 and stands on 6-7. If the player
    ///   drew, the banker draws according to the standard banker drawing table.
    ///   If the player stood, the banker draws independently on 0-5.
    ///
    /// Returns `None` after the cut card has been dealt and the final hand
    /// played out.
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted {
            return None;
        }
        let mut player = BaccaratHand::default();
        let mut banker = BaccaratHand::default();
        if self.shoe.has_reached_cut_card() {
            self.is_exhausted = true;
        }
        player.take(&self.pull());
        banker.take(&self.pull());
        player.take(&self.pull());
        banker.take(&self.pull());
        if no_natural(&player, &banker) {
            if !stand_pat(&player) {
                let player_third = self.pull();
                player.take(&player_third);
                if banker_take_third(&banker, player_third) {
                    banker.take(&self.pull());
                }
            } else if !stand_pat(&banker) {
                banker.take(&self.pull());
            }
        }
        Some(Self::Item { player, banker })
    }
}

#[cfg(test)]
fn hand(cards: &[kev::CardInt]) -> BaccaratHand {
    let mut hand: BaccaratHand = BaccaratHand::default();
    cards.iter().for_each(|c| hand.take(c));
    hand
}

#[cfg(test)]
mod pip_value_tests {
    use super::pip_value;
    use kev::CardInt;
    use rstest::rstest;

    #[rstest]
    #[case(CardInt::CardAs, 1)]
    #[case(CardInt::CardKs, 10)]
    #[case(CardInt::CardQs, 10)]
    #[case(CardInt::CardJs, 10)]
    #[case(CardInt::CardTs, 10)]
    #[case(CardInt::Card9s, 9)]
    #[case(CardInt::Card8s, 8)]
    #[case(CardInt::Card7s, 7)]
    #[case(CardInt::Card6s, 6)]
    #[case(CardInt::Card5s, 5)]
    #[case(CardInt::Card4s, 4)]
    #[case(CardInt::Card3s, 3)]
    #[case(CardInt::Card2s, 2)]
    fn pip_value_for_rank(#[case] card: CardInt, #[case] expected: u8) {
        assert_eq!(pip_value(card), expected);
    }
}

#[cfg(test)]
mod no_natural_tests {
    use super::no_natural;
    use kev::CardInt;
    use rstest::rstest;

    #[rstest]
    // neither natural -> true
    #[case(&[CardInt::Card3s, CardInt::Card4h], &[CardInt::Card2s, CardInt::Card5h], true)]
    // player natural -> false
    #[case(&[CardInt::Card4s, CardInt::Card4h], &[CardInt::Card2s, CardInt::Card3h], false)]
    // banker natural -> false
    #[case(&[CardInt::Card2s, CardInt::Card3h], &[CardInt::Card4s, CardInt::Card5h], false)]
    // both naturals -> false
    #[case(&[CardInt::Card4s, CardInt::Card4h], &[CardInt::Card4d, CardInt::Card5c], false)]
    fn no_natural_cases(
        #[case] player: &[CardInt],
        #[case] banker: &[CardInt],
        #[case] expected: bool,
    ) {
        assert_eq!(
            no_natural(&super::hand(player), &super::hand(banker)),
            expected
        );
    }

    // panics if player hand already has a third card
    #[test]
    #[should_panic]
    fn panics_if_player_has_third() {
        let p = super::hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        let b = super::hand(&[CardInt::Card4s, CardInt::Card5h]);
        no_natural(&p, &b);
    }

    // panics if banker hand already has a third card
    #[test]
    #[should_panic]
    fn panics_if_banker_has_third() {
        let p = super::hand(&[CardInt::Card4s, CardInt::Card5h]);
        let b = super::hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        no_natural(&p, &b);
    }
}

#[cfg(test)]
mod stand_pat_tests {
    use super::stand_pat;
    use kev::CardInt;
    use rstest::rstest;

    #[rstest]
    #[case(&[CardInt::Card3s, CardInt::Card3h], false)] // value=6: stands
    #[case(&[CardInt::Card3s, CardInt::Card4h], false)] // value=7: stands
    #[case(&[CardInt::CardKs, CardInt::Card5h], true)] // value=5: draws
    #[case(&[CardInt::Card2s, CardInt::Card3h], true)] // value=5: draws
    #[case(&[CardInt::CardKs, CardInt::CardKh], true)] // value=0: draws
    fn stand_pat_cases(#[case] cards: &[CardInt], #[case] draws: bool) {
        assert_eq!(stand_pat(&super::hand(cards)), !draws);
    }

    #[test]
    #[should_panic]
    fn panics_if_hand_has_third() {
        let h = super::hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        stand_pat(&h);
    }
}

#[cfg(test)]
mod banker_take_third_tests {
    use super::banker_take_third;
    use kev::CardInt;
    use rstest::rstest;

    // score 0-2: always draws regardless of player third card
    #[rstest]
    #[case(&[CardInt::CardKs, CardInt::CardKh], CardInt::CardAs)] // value=0
    #[case(&[CardInt::CardKs, CardInt::CardAh], CardInt::Card8s)] // value=1
    #[case(&[CardInt::CardKs, CardInt::Card2h], CardInt::Card8s)] // value=2
    fn score_0_to_2_always_draws(#[case] banker: &[CardInt], #[case] player_third: CardInt) {
        assert!(banker_take_third(&super::hand(banker), player_third));
    }

    // score 3: draws on any pip except 8
    #[rstest]
    #[case(CardInt::CardAs, true)] // pip=1
    #[case(CardInt::Card2s, true)] // pip=2
    #[case(CardInt::Card7s, true)] // pip=7
    #[case(CardInt::Card9s, true)] // pip=9
    #[case(CardInt::CardKs, true)] // pip=10
    #[case(CardInt::Card8s, false)] // pip=8: does not draw
    fn score_3(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = super::hand(&[CardInt::CardAs, CardInt::Card2h]); // value=3
        assert_eq!(banker_take_third(&banker, player_third), expected);
    }

    // score 4: draws on pip 2-7
    #[rstest]
    #[case(CardInt::Card2s, true)] // pip=2: boundary low
    #[case(CardInt::Card5s, true)] // pip=5: mid
    #[case(CardInt::Card7s, true)] // pip=7: boundary high
    #[case(CardInt::CardAs, false)] // pip=1: below range
    #[case(CardInt::Card8s, false)] // pip=8: above range
    fn score_4(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = super::hand(&[CardInt::Card2s, CardInt::Card2h]); // value=4
        assert_eq!(banker_take_third(&banker, player_third), expected);
    }

    // score 5: draws on pip 4-7
    #[rstest]
    #[case(CardInt::Card4s, true)] // pip=4: boundary low
    #[case(CardInt::Card6s, true)] // pip=6: mid
    #[case(CardInt::Card7s, true)] // pip=7: boundary high
    #[case(CardInt::Card3s, false)] // pip=3: below range
    #[case(CardInt::Card8s, false)] // pip=8: above range
    fn score_5(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = super::hand(&[CardInt::Card2s, CardInt::Card3h]); // value=5
        assert_eq!(banker_take_third(&banker, player_third), expected);
    }

    // score 6: draws on pip 6-7 (covered by iterator tests; boundary cases added here)
    #[rstest]
    #[case(CardInt::Card6s, true)] // pip=6: boundary low
    #[case(CardInt::Card7s, true)] // pip=7: boundary high
    #[case(CardInt::Card5s, false)] // pip=5: below range
    #[case(CardInt::Card8s, false)] // pip=8: above range
    fn score_6(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = super::hand(&[CardInt::Card3s, CardInt::Card3h]); // value=6
        assert_eq!(banker_take_third(&banker, player_third), expected);
    }

    // score 7: never draws
    #[rstest]
    #[case(CardInt::CardAs)]
    #[case(CardInt::Card8s)]
    #[case(CardInt::CardKs)]
    fn score_7_never_draws(#[case] player_third: CardInt) {
        let banker = super::hand(&[CardInt::Card3s, CardInt::Card4h]); // value=7
        assert!(!banker_take_third(&banker, player_third));
    }

    // panic if banker hand already has a third card
    #[test]
    #[should_panic]
    fn panics_if_banker_has_third() {
        let banker = super::hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        banker_take_third(&banker, CardInt::Card4s);
    }
}

#[cfg(test)]
mod baccarat_hand_tests {
    use kev::CardInt;
    use rstest::rstest;

    #[rstest]
    // empty
    #[case(&[], 0)]
    // single card - each rank
    #[case(&[CardInt::CardAs], 1)]
    #[case(&[CardInt::CardKs], 0)]
    #[case(&[CardInt::CardQs], 0)]
    #[case(&[CardInt::CardJs], 0)]
    #[case(&[CardInt::CardTs], 0)]
    #[case(&[CardInt::Card9s], 9)]
    #[case(&[CardInt::Card8s], 8)]
    #[case(&[CardInt::Card7s], 7)]
    #[case(&[CardInt::Card6s], 6)]
    #[case(&[CardInt::Card5s], 5)]
    #[case(&[CardInt::Card4s], 4)]
    #[case(&[CardInt::Card3s], 3)]
    #[case(&[CardInt::Card2s], 2)]
    // suit does not affect value
    #[case(&[CardInt::Card9h], 9)]
    #[case(&[CardInt::Card9d], 9)]
    #[case(&[CardInt::Card9c], 9)]
    // two-card sums
    #[case(&[CardInt::Card5s, CardInt::Card4h], 9)]
    #[case(&[CardInt::CardKs, CardInt::Card9h], 9)]
    #[case(&[CardInt::Card6s, CardInt::Card2h], 8)]
    // three-card sums
    #[case(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d], 6)]
    #[case(&[CardInt::CardKs, CardInt::CardQh, CardInt::Card7d], 7)]
    // mod 10 reduction: two-card sums exceeding 9
    #[case(&[CardInt::Card5s, CardInt::Card5h], 0)] // 10 % 10 = 0
    #[case(&[CardInt::Card7s, CardInt::Card6h], 3)] // 13 % 10 = 3
    #[case(&[CardInt::Card9s, CardInt::Card8h], 7)] // 17 % 10 = 7
    #[case(&[CardInt::Card9s, CardInt::Card9h], 8)] // 18 % 10 = 8
    // mod 10 reduction: three-card sums exceeding 9
    #[case(&[CardInt::Card9s, CardInt::Card9h, CardInt::Card9d], 7)] // 27 % 10 = 7
    fn value(#[case] cards: &[CardInt], #[case] expected: u8) {
        assert_eq!(super::hand(cards).value(), expected);
    }

    // --- is_pair ---

    #[rstest]
    #[case(&[CardInt::CardAs, CardInt::CardAh], true)]
    #[case(&[CardInt::Card9s, CardInt::Card9d], true)]
    #[case(&[CardInt::Card9s, CardInt::Card9d, CardInt::CardAs], true)]
    #[case(&[CardInt::CardAs, CardInt::CardKs], false)]
    #[case(&[CardInt::Card9s, CardInt::Card8h], false)]
    #[case(&[CardInt::Card9s, CardInt::Card8h, CardInt::Card8d], false)]
    fn is_pair(#[case] cards: &[CardInt], #[case] expected: bool) {
        assert_eq!(super::hand(cards).is_pair(), expected);
    }

    // --- has_third ---

    #[rstest]
    #[case(&[CardInt::CardAs], false)]
    #[case(&[CardInt::CardAs, CardInt::CardKs], false)]
    #[case(&[CardInt::CardAs, CardInt::CardKs, CardInt::CardQs], true)]
    fn has_third(#[case] cards: &[CardInt], #[case] expected: bool) {
        assert_eq!(super::hand(cards).has_third(), expected);
    }
}

#[cfg(test)]
mod baccarat_round_tests {
    use super::BaccaratRound;
    use kev::CardInt;
    use rstest::rstest;

    fn round(player: &[CardInt], banker: &[CardInt]) -> BaccaratRound {
        BaccaratRound {
            player: super::hand(player),
            banker: super::hand(banker),
        }
    }

    #[rstest]
    // outcome: player wins (9 vs 0), no pairs, no thirds
    #[case(&[CardInt::Card9s, CardInt::CardKh], &[CardInt::CardKs, CardInt::CardTh], 0x0901)]
    // outcome: banker wins (0 vs 9), no pairs, no thirds
    #[case(&[CardInt::CardKs, CardInt::CardTh], &[CardInt::Card9s, CardInt::CardKh], 0x9002)]
    // outcome: tie (5 vs 5), no pairs, no thirds
    #[case(&[CardInt::Card5s, CardInt::CardKh], &[CardInt::Card5h, CardInt::CardQd], 0x5503)]
    // player pair: bit 2 set (A==A), player wins (2 vs 0)
    #[case(&[CardInt::CardAs, CardInt::CardAh], &[CardInt::CardKs, CardInt::CardTh], 0x0205)]
    // banker pair: bit 3 set (A==A), banker wins (0 vs 2)
    #[case(&[CardInt::CardKs, CardInt::CardTh], &[CardInt::CardAs, CardInt::CardAh], 0x200A)]
    // player has third: bit 4 set, player wins (6 vs 5)
    #[case(&[CardInt::Card6s, CardInt::CardKh, CardInt::CardTs], &[CardInt::Card5s, CardInt::CardQd], 0x5611)]
    // banker has third: bit 5 set, banker wins (5 vs 6)
    #[case(&[CardInt::Card5s, CardInt::CardQd], &[CardInt::Card6s, CardInt::CardKh, CardInt::CardTs], 0x6522)]
    // both thirds: bits 4+5 set, player wins (9 vs 2)
    #[case(&[CardInt::Card2h, CardInt::Card3d, CardInt::Card4c], &[CardInt::Card3h, CardInt::Card4d, CardInt::Card5c], 0x2931)]
    // both pairs: bits 2+3 set, player wins (2 vs 0)
    #[case(&[CardInt::CardAs, CardInt::CardAh], &[CardInt::CardKs, CardInt::CardKh], 0x020D)]
    fn onehot(#[case] player: &[CardInt], #[case] banker: &[CardInt], #[case] expected: u32) {
        assert_eq!(round(player, banker).onehot(), expected);
    }

    #[test]
    #[should_panic]
    fn onehot_panics_if_player_has_fewer_than_two_cards() {
        let _ = round(&[CardInt::CardAs], &[CardInt::CardKs, CardInt::CardKh]).onehot();
    }
}

#[cfg(test)]
mod baccarat_shoe_tests {
    use super::{BaccaratScoreboard, BaccaratShoe};
    use kev::CardInt;
    use num_bigint::BigUint;
    use shoe::Card;

    /// Builds a shoe vec where `first` is dealt first and `rest` are the
    /// remaining cards available to burn. Layout (dealt right-to-left):
    ///
    /// ```text
    /// [Cut, rest[0], rest[1], ..., first]
    /// ```
    ///
    /// `cut_pos = 0`, so `burn(n)` requires `n <= rest.len()`.
    fn shoe_vec(first: CardInt, rest: &[CardInt]) -> Vec<Card> {
        let mut cards = vec![Card::Cut];
        cards.extend(rest.iter().map(|&c| Card::Play(c)));
        cards.push(Card::Play(first));
        cards
    }

    #[test]
    fn ace_burns_one_card() {
        // Ace has value 1 -> burns 1 card; next card out of the shoe must be CardKs.
        // shoe_vec layout: [Cut, Ks, 2s, As] - 2s is burned, Ks is next.
        let cards = shoe_vec(CardInt::CardAs, &[CardInt::CardKs, CardInt::Card2s]);
        let mut shoe = BaccaratShoe::from(cards);
        assert_eq!(shoe.pull(), CardInt::CardKs);
    }

    #[test]
    fn king_burns_ten_cards() {
        // King has value 10 -> burns 10 cards; next card out of the shoe must be CardAs.
        // shoe_vec layout: [Cut, As, 2s..Ts, Qs, Ks] - 2s..Qs are burned, As is next.
        let rest = [
            CardInt::CardAs,
            CardInt::Card2s,
            CardInt::Card3s,
            CardInt::Card4s,
            CardInt::Card5s,
            CardInt::Card6s,
            CardInt::Card7s,
            CardInt::Card8s,
            CardInt::Card9s,
            CardInt::CardTs,
            CardInt::CardQs,
        ];
        let cards = shoe_vec(CardInt::CardKs, &rest);
        let mut shoe = BaccaratShoe::from(cards);
        assert_eq!(shoe.pull(), CardInt::CardAs);
    }

    #[test]
    #[should_panic(expected = "cut card dealt as first card")]
    fn cut_card_first_panics() {
        // Cut at the last position -> dealt as the first card -> unreachable! branch.
        let cards = vec![Card::Play(CardInt::CardAs), Card::Cut];
        let _ = BaccaratShoe::from(cards);
    }

    #[test]
    #[should_panic(
        expected = "pen leaves fewer than 12 cards after the cut card for a 1-deck shoe"
    )]
    fn new_pen_too_high_panics() {
        // 1-deck shoe has 52 cards; cut_pos = floor((1.0 - 0.8) * 52) = 10 < 12.
        let _ = BaccaratShoe::new(1, 1, 0.8);
    }

    #[test]
    #[should_panic(expected = "burning too many cards")]
    fn insufficient_cards_to_burn_panics() {
        // King requires burning 10 cards, but only 5 are available after it.
        let rest = [
            CardInt::CardAs,
            CardInt::Card2s,
            CardInt::Card3s,
            CardInt::Card4s,
            CardInt::Card5s,
        ];
        let cards = shoe_vec(CardInt::CardKs, &rest);
        let _ = BaccaratShoe::from(cards);
    }

    // -- Iterator trait tests -------------------------------------------------
    //
    // Each test builds a shoe via BaccaratShoe::from(Vec<Card>), drives one
    // round through the Iterator trait, and asserts that player_cards() and
    // banker_cards() match exactly the cards placed into the shoe.
    //
    // shoe_vec layout: [Cut, rest[0], ..., rest[n-1], first_card]
    // Deal order (end-first): first_card -> rest[n-1] -> rest[n-2] -> ... -> rest[0]
    // first_card pip value determines burn count (rest[n-1..n-1-pip] are burned).
    //
    // With first=As (pip=1, burns rest[n-1]) the play sequence is:
    //   player card 1 = rest[n-2], banker card 1 = rest[n-3],
    //   player card 2 = rest[n-4], banker card 2 = rest[n-5], ...

    // -- Iterator trait + cut-card exhaustion tests --------------------------
    //
    // Vec layout (dealer reads from the end; cards[0] is never dealt):
    //
    //   [ dummy | ...play cards (deepest->shallowest)... | Cut | burn | As ]
    //    index 0                                           L-3   L-2    L-1
    //
    // Shoe::from sets cursor = L-1 and cut_pos = index of Card::Cut = L-3.
    // from_shoe: deal As (cursor -> L-2), burn(1) (cursor -> L-3).
    // has_reached_cut_card() = (cursor <= cut_pos) = (L-3 <= L-3) = true.
    // next() call 1: sees has_reached_cut_card()=true -> sets is_exhausted=true,
    //   skips the Cut via pull's loop, deals all play cards, returns Some.
    // next() call 2: is_exhausted=true -> returns None.

    #[test]
    fn iterator_natural_no_third_cards() {
        // player=[Card9s,CardKs] value=9 (natural) -> no thirds drawn
        // banker=[Card5h,Card2d] value=7
        let burn = CardInt::Card3c;
        let c0 = CardInt::Card2d; // banker card 2
        let c1 = CardInt::CardKs; // player card 2
        let c2 = CardInt::Card5h; // banker card 1
        let c3 = CardInt::Card9s; // player card 1 (natural)
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(c0),
            Card::Play(c1),
            Card::Play(c2),
            Card::Play(c3),
            Card::Cut, // right after burn card in deal sequence
            Card::Play(burn),
            Card::Play(CardInt::CardAs),
        ];
        let mut shoe = BaccaratShoe::from(cards);
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[c3, c1]);
        assert_eq!(round.banker_cards(), &[c2, c0]);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn iterator_player_stands_banker_draws_third() {
        // player=[Card6s,CardKh] value=6 (stand_pat) -> player stands
        // banker=[Card5h,CardKd] value=5 (!stand_pat) -> banker draws r0
        let burn = CardInt::Card3c;
        let r0 = CardInt::Card7d; // banker card 3
        let r1 = CardInt::CardKd; // banker card 2
        let r2 = CardInt::CardKh; // player card 2
        let r3 = CardInt::Card5h; // banker card 1
        let r4 = CardInt::Card6s; // player card 1
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(r0),
            Card::Play(r1),
            Card::Play(r2),
            Card::Play(r3),
            Card::Play(r4),
            Card::Cut, // right after burn card in deal sequence
            Card::Play(burn),
            Card::Play(CardInt::CardAs),
        ];
        let mut shoe = BaccaratShoe::from(cards);
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[r4, r2]);
        assert_eq!(round.banker_cards(), &[r3, r1, r0]);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn iterator_player_draws_banker_stands_on_seven() {
        // player=[Card2s,Card3s] value=5 (!stand_pat) -> player draws r0=Card5c (pip=5)
        // banker=[Card3h,Card4d] value=7 -> banker_take_third(7, 5c): _ => false -> stands
        let burn = CardInt::Card9c;
        let r0 = CardInt::Card5c; // player card 3
        let r1 = CardInt::Card4d; // banker card 2
        let r2 = CardInt::Card3s; // player card 2
        let r3 = CardInt::Card3h; // banker card 1
        let r4 = CardInt::Card2s; // player card 1
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(r0),
            Card::Play(r1),
            Card::Play(r2),
            Card::Play(r3),
            Card::Play(r4),
            Card::Cut, // right after burn card in deal sequence
            Card::Play(burn),
            Card::Play(CardInt::CardAs),
        ];
        let mut shoe = BaccaratShoe::from(cards);
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[r4, r2, r0]);
        assert_eq!(round.banker_cards(), &[r3, r1]);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn iterator_both_draw_third_cards() {
        // player=[Card2h,Card2d] value=4 (!stand_pat) -> draws r1=Card6d (pip=6)
        // banker=[Card2c,Card4d] value=6 -> banker_take_third(6, 6d): pip=6 matches(6|7) -> draws r0
        let burn = CardInt::Card3c;
        let r0 = CardInt::Card7c; // banker card 3
        let r1 = CardInt::Card6d; // player card 3 (pip=6)
        let r2 = CardInt::Card4d; // banker card 2
        let r3 = CardInt::Card2d; // player card 2
        let r4 = CardInt::Card2c; // banker card 1
        let r5 = CardInt::Card2h; // player card 1
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(r0),
            Card::Play(r1),
            Card::Play(r2),
            Card::Play(r3),
            Card::Play(r4),
            Card::Play(r5),
            Card::Cut, // right after burn card in deal sequence
            Card::Play(burn),
            Card::Play(CardInt::CardAs),
        ];
        let mut shoe = BaccaratShoe::from(cards);
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[r5, r3, r1]);
        assert_eq!(round.banker_cards(), &[r4, r2, r0]);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn iterator_cut_card_mid_first_round_exhausts_after_second() {
        // Vec layout (L=12, dealt right-to-left):
        //
        //   idx: [ 0    |  1    2    3    4   |  5    6   |  7  |. 8    9   |  10  11]
        //        [dummy | r2b2 r2p2 r2b1 r2p1 | r1b2 r1p2 | Cut | r1b1 r1p1 | burn As]
        //
        // cursor starts=11, cut_pos=7.
        // from_shoe: deal As (cursor->10), burn(1) (cursor->9). 9>7 -> not exhausted yet.
        //
        // Round 1 next(): has_reached_cut_card()=false (9>7).
        //   pull p1 : deals cards[9]=Card9s              cursor->8
        //   pull b1 : deals cards[8]=Card5h              cursor->7
        //   pull p2 : deals cards[7]=Card::Cut -> skip;   cursor->6
        //             deals cards[6]=CardKs               cursor->5  -> p2=CardKs
        //   pull b2 : deals cards[5]=Card2d              cursor->4
        //   player=[Card9s,CardKs] value=9 (natural) -> no thirds.
        //   Returns Some(round1). After return: cursor=4 <= cut_pos=7 -> exhausted on next check.
        //
        // Round 2 next(): has_reached_cut_card()=true (4<=7) -> is_exhausted=true.
        //   pull p1 : cards[4]=Card2h  cursor->3
        //   pull b1 : cards[3]=Card9c  cursor->2
        //   pull p2 : cards[2]=Card5c  cursor->1
        //   pull b2 : cards[1]=CardKd  cursor->0
        //   player=[Card2h,Card5c] value=7, banker=[Card9c,CardKd] value=9 (natural) -> no thirds.
        //   Returns Some(round2).
        //
        // Round 3 next(): is_exhausted=true -> None.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::CardKd), // round 2 banker card 2
            Card::Play(CardInt::Card5c), // round 2 player card 2
            Card::Play(CardInt::Card9c), // round 2 banker card 1
            Card::Play(CardInt::Card2h), // round 2 player card 1
            Card::Play(CardInt::Card2d), // round 1 banker card 2
            Card::Play(CardInt::CardKs), // round 1 player card 2
            Card::Cut,                   // cut card - encountered during round 1's p2 pull
            Card::Play(CardInt::Card5h), // round 1 banker card 1
            Card::Play(CardInt::Card9s), // round 1 player card 1
            Card::Play(CardInt::Card3c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccaratShoe::from(cards);

        let round1 = shoe.next().expect("round 1 should be dealt");
        assert_eq!(round1.player_cards(), &[CardInt::Card9s, CardInt::CardKs]);
        assert_eq!(round1.banker_cards(), &[CardInt::Card5h, CardInt::Card2d]);

        let round2 = shoe.next().expect("round 2 should be dealt");
        assert_eq!(round2.player_cards(), &[CardInt::Card2h, CardInt::Card5c]);
        assert_eq!(round2.banker_cards(), &[CardInt::Card9c, CardInt::CardKd]);

        assert!(shoe.next().is_none());
    }

    #[test]
    fn all_scoreboards_accumulate_correctly_over_12_rounds() {
        // first_card = As (pip=1, burns 1 card: Card2h).
        // Round  1: P=[9d, Qh]     value=9 natural, B=[9c, Ts]     value=9 natural -> tie,         bead=0x93.
        // Round  2: P=[3c, Kd, 8c] value=1,         B=[6s, Jh]     value=6         -> banker wins, bead=0x62.
        // Round  3: P=[5d, 7c]     value=2,         B=[9h, Tc]     value=9 natural -> banker wins, bead=0x92.
        // Round  4: P=[Qs, 9d]     value=9 natural, B=[4s, 5s]     value=9 natural -> tie,         bead=0x93.
        // Round  5: P=[Ac, 6s]     value=7,         B=[7h, Kc]     value=7         -> tie,         bead=0x73.
        // Round  6: P=[Ah, 7s]     value=8 natural, B=[Ad, 6c]     value=7         -> player wins, bead=0x81.
        // Round  7: P=[6h, Qd]     value=6,         B=[2c, Kh, Ts] value=2         -> player wins, bead=0x61.
        // Round  8: P=[Ks, 9c]     value=9 natural, B=[8h, 7d]     value=5         -> player wins, bead=0x91.
        // Round  9: P=[9s, 2d]     value=1,         B=[8s, Tc]     value=8 natural -> banker wins, bead=0x82.
        // Round 10: P=[9h, Jd]     value=9 natural, B=[4d, 6d]     value=0         -> player wins, bead=0x91.
        // Round 11: P=[3s, Th, Js] value=3,         B=[9s, 8d]     value=7         -> banker wins, bead=0x72.
        // Round 12: P=[4h, Qc, 3h] value=7,         B=[Td, 3d, 2s] value=5         -> player wins, bead=0x71.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            // round 12 cards
            Card::Play(CardInt::Card2s), // round 12 banker card 3  (position 54)
            Card::Play(CardInt::Card3h), // round 12 player card 3  (position 53)
            Card::Play(CardInt::Card3d), // round 12 banker card 2  (position 52)
            Card::Play(CardInt::CardQc), // round 12 player card 2  (position 51)
            Card::Play(CardInt::CardTd), // round 12 banker card 1  (position 50)
            Card::Play(CardInt::Card4h), // round 12 player card 1  (position 49)
            // round 11 cards
            Card::Play(CardInt::CardJs), // round 11 player card 3  (position 48)
            Card::Play(CardInt::Card8d), // round 11 banker card 2  (position 47)
            Card::Cut,                   // cut card between positions 46 and 47
            Card::Play(CardInt::CardTh), // round 11 player card 2  (position 46)
            Card::Play(CardInt::Card9s), // round 11 banker card 1  (position 45)
            Card::Play(CardInt::Card3s), // round 11 player card 1  (position 44)
            // round 10 cards
            Card::Play(CardInt::Card6d), // round 10 banker card 2  (position 43)
            Card::Play(CardInt::CardJd), // round 10 player card 2  (position 42)
            Card::Play(CardInt::Card4d), // round 10 banker card 1  (position 41)
            Card::Play(CardInt::Card9h), // round 10 player card 1  (position 40)
            // round 9 cards
            Card::Play(CardInt::CardTc), // round 9 banker card 2   (position 39)
            Card::Play(CardInt::Card2d), // round 9 player card 2   (position 38)
            Card::Play(CardInt::Card8s), // round 9 banker card 1   (position 37)
            Card::Play(CardInt::Card9s), // round 9 player card 1   (position 36)
            // round 8 cards
            Card::Play(CardInt::Card7d), // round 8 banker card 2   (position 35)
            Card::Play(CardInt::Card9c), // round 8 player card 2   (position 34)
            Card::Play(CardInt::Card8h), // round 8 banker card 1   (position 33)
            Card::Play(CardInt::CardKs), // round 8 player card 1   (position 32)
            // round 7 cards
            Card::Play(CardInt::CardTs), // round 7 banker card 3   (position 31)
            Card::Play(CardInt::CardKh), // round 7 banker card 2   (position 30)
            Card::Play(CardInt::CardQd), // round 7 player card 2   (position 29)
            Card::Play(CardInt::Card2c), // round 7 banker card 1   (position 28)
            Card::Play(CardInt::Card6h), // round 7 player card 1   (position 27)
            // round 6 cards
            Card::Play(CardInt::Card6c), // round 6 banker card 2   (position 26)
            Card::Play(CardInt::Card7s), // round 6 player card 2   (position 25)
            Card::Play(CardInt::CardAd), // round 6 banker card 1   (position 24)
            Card::Play(CardInt::CardAh), // round 6 player card 1   (position 23)
            // round 5 cards
            Card::Play(CardInt::CardKc), // round 5 banker card 2   (position 22)
            Card::Play(CardInt::Card6s), // round 5 player card 2   (position 21)
            Card::Play(CardInt::Card7h), // round 5 banker card 1   (position 20)
            Card::Play(CardInt::CardAc), // round 5 player card 1   (position 19)
            // round 4 cards
            Card::Play(CardInt::Card5s), // round 4 banker card 2   (position 18)
            Card::Play(CardInt::Card9d), // round 4 player card 2   (position 17)
            Card::Play(CardInt::Card4s), // round 4 banker card 1   (position 16)
            Card::Play(CardInt::CardQs), // round 4 player card 1   (position 15)
            // round 3 cards
            Card::Play(CardInt::CardTc), // round 3 banker card 2   (position 14)
            Card::Play(CardInt::Card7c), // round 3 player card 2   (position 13)
            Card::Play(CardInt::Card9h), // round 3 banker card 1   (position 12)
            Card::Play(CardInt::Card5d), // round 3 player card 1   (position 11)
            // round 2 cards
            Card::Play(CardInt::Card8c), // round 2 player card 3  (position 10)
            Card::Play(CardInt::CardJh), // round 2 banker card 2  (position  9)
            Card::Play(CardInt::CardKd), // round 2 player card 2  (position  8)
            Card::Play(CardInt::Card6s), // round 2 banker card 1  (position  7)
            Card::Play(CardInt::Card3c), // round 2 player card 1  (position  6)
            // round 1 cards
            Card::Play(CardInt::CardTs), // round 1 banker card 2  (position  5)
            Card::Play(CardInt::CardQh), // round 1 player card 2  (position  4)
            Card::Play(CardInt::Card9c), // round 1 banker card 1  (position  3)
            Card::Play(CardInt::Card9d), // round 1 player card 1  (position  2)
            Card::Play(CardInt::Card2h), // burn card               (position  1)
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1) (position 0)
        ];
        let mut sb = BaccaratScoreboard::new();
        let mut shoe = BaccaratShoe::from(cards);
        for _ in 0..2 {
            let round = shoe.next().expect("round should be dealt");
            sb.update(&round);
        }
        // bead_plate = (round1_bead << 8) | round2_bead = (0x93 << 8) | 0x62 = 0x9362 = 37730.
        assert_eq!(*sb.bead_plate(), BigUint::from(37730u32));
        assert_eq!(*sb.big_road(), BigUint::from(90625u32));
        for _ in 0..10 {
            let round = shoe.next().expect("round should be dealt");
            sb.update(&round);
        }
        assert_eq!(
            *sb.bead_plate(),
            BigUint::parse_bytes(b"936292937381619182917271", 16).expect("valid bead_plate hex")
        );
        assert_eq!(
            *sb.big_road(),
            BigUint::parse_bytes(b"016202920200810061009103008201009101007201007101", 16)
                .expect("valid big_road hex")
        );
        assert_eq!(
            *sb.derived_roads(),
            [
                BigUint::parse_bytes(b"030605", 16).expect("valid big_eye_boy hex"),
                BigUint::parse_bytes(b"0403", 16).expect("valid small_road hex"),
                BigUint::parse_bytes(b"04", 16).expect("valid cockroach_pig hex"),
            ]
        );
        sb.clear();
        assert_eq!(*sb.bead_plate(), BigUint::ZERO);
        assert_eq!(*sb.big_road(), BigUint::ZERO);
        assert_eq!(
            *sb.derived_roads(),
            [BigUint::ZERO, BigUint::ZERO, BigUint::ZERO]
        );
    }
}
