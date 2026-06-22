//! Baccarat engine library.
//!
//! `bacc` deals baccarat rounds from a multi-deck shoe and maintains the five
//! standard scoreboards as compact byte sequences.
//!
//! # Core types
//!
//! - [`BaccShoe`] - wraps `shoe::Shoe` with baccarat dealing rules. Implements
//!   [`Iterator`]`<Item = `[`BaccRound`]`>` so rounds can be consumed with
//!   a plain `for` loop.
//! - [`BaccScoreboard`] - tracks the five scoreboards.
//! - [`BaccRound`] - the resolved outcome of one round. Exposes the player
//!   and banker card slices, a compact [`BaccRound::encode`] encoding of the
//!   full card sequence and metadata, and a [`BaccRound::describe`] TOML
//!   fragment for human-readable output.
//! - [`BaccHand`] - the cards dealt to one side, with hand value calculation,
//!   pair detection, and third-card flag.
//!
//! # Scoreboards
//!
//! All five scoreboards are stored as byte sequences. Call
//! [`BaccScoreboard::update`] after each round to advance them:
//!
//! - **Serialization** ([`BaccScoreboard::encode`] / [`BaccScoreboard::decode`]) -
//!   encodes the full scoreboard state as a hex string of bead words, sufficient
//!   to reconstruct all five roads.
//! - **Bead plate** ([`BaccScoreboard::simulate_bead_plate`]) - returns a
//!   column-major grid of `(bead_byte, aux_byte)` cells.
//! - **Big road** ([`BaccScoreboard::simulate_big_road`]) - returns a
//!   [`ROWS`]-high grid of at most [`MAX_COL_COUNT`] columns.
//! - **Derived roads** ([`BaccScoreboard::simulate_derived_road`]) - Big Eye Boy
//!   (index 0), Small Road (index 1), and Cockroach Pig (index 2), one grid each.
//!
//! # Usage
//!
//! ```rust
//! use bacc::{BaccScoreboard, BaccShoe};
//! use shoe::{Card, Shoe, DECK};
//!
//! let mut cards: Vec<Card> = (0..8).flat_map(|_| DECK).collect();
//! cards.push(Card::Cut);
//! let last = cards.len() - 1;
//! cards.swap(last, 14);
//!
//! let mut sb = BaccScoreboard::new();
//! let shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
//! for round in shoe {
//!     sb.update(&round);
//!     print!("{}", round.describe());
//!     println!("[scoreboard]");
//!     println!("encoded = \"{}\"", sb.encode());
//! }
//! ```

use bacc_core::pip_value;
use kev::CardInt;
use shoe::{Card, Shoe};

pub use bacc_core::{BaccHand, BaccOutcome, BaccRound, BaccScoreboard, MAX_COL_COUNT, ROWS};

/// A baccarat shoe that deals [`BaccRound`]s via the [`Iterator`] trait.
///
/// Wraps `shoe::Shoe` with baccarat-specific dealing: the burn ritual is
/// applied on construction (first card exposed, that many cards discarded),
/// and each round is dealt using the standard third-card rules.
pub struct BaccShoe {
    shoe: shoe::Shoe,
    // Set after the cut card is reached; the current hand plays out and then the iterator stops.
    is_exhausted: bool,
}

impl BaccShoe {
    fn deal(&mut self) -> (CardInt, bool) {
        let mut saw_cut = false;
        loop {
            match self.shoe.deal().expect("shoe is non-empty") {
                Card::Play(c) => break (c, saw_cut),
                Card::Cut => saw_cut = true,
            }
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
    fn no_natural(p: &BaccHand, b: &BaccHand) -> bool {
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
    fn stand_pat(hand: &BaccHand) -> bool {
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
    fn banker_take_third(banker_hand: &BaccHand, player_third_card: CardInt) -> bool {
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
}

impl From<Shoe> for BaccShoe {
    fn from(mut shoe: Shoe) -> Self {
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
}

impl Iterator for BaccShoe {
    type Item = BaccRound;

    /// Deals and returns the next [`BaccRound`], or `None` when the shoe
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
        let mut player = BaccHand::default();
        let mut banker = BaccHand::default();
        let mut banker_forced_third = false;
        let mut cut_card_index: Option<u8> = None;
        let mut card_index: u8 = 0;
        if self.shoe.has_reached_cut_card() {
            self.is_exhausted = true;
        }
        let mut deal = |shoe: &mut Self| {
            let (card, saw_cut) = shoe.deal();
            if saw_cut {
                cut_card_index = Some(card_index);
            }
            card_index += 1;
            card
        };
        player.take(&deal(self));
        banker.take(&deal(self));
        player.take(&deal(self));
        banker.take(&deal(self));
        if Self::no_natural(&player, &banker) {
            if !Self::stand_pat(&player) {
                let player_third = deal(self);
                player.take(&player_third);
                if Self::banker_take_third(&banker, player_third) {
                    banker_forced_third = banker.value() <= 2;
                    banker.take(&deal(self));
                }
            } else if !Self::stand_pat(&banker) {
                banker.take(&deal(self));
            }
        }
        Some(BaccRound::new(
            player,
            banker,
            banker_forced_third,
            cut_card_index,
        ))
    }
}

#[cfg(test)]
mod baccarat_shoe_tests {
    use super::BaccShoe;
    use kev::CardInt;
    use rstest::rstest;
    use shoe::{Card, Shoe};

    fn hand(cards: &[CardInt]) -> super::BaccHand {
        let mut h = super::BaccHand::default();
        cards.iter().for_each(|c| h.take(c));
        h
    }

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
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        assert_eq!(shoe.deal(), (CardInt::CardKs, false));
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
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        assert_eq!(shoe.deal(), (CardInt::CardAs, false));
    }

    #[test]
    #[should_panic(expected = "cut card dealt as first card")]
    fn cut_card_first_panics() {
        // Cut at the last position -> dealt as the first card -> unreachable! branch.
        let cards = vec![Card::Play(CardInt::CardAs), Card::Cut];
        let _ = BaccShoe::from(Shoe::from(cards.as_slice()));
    }

    // -- no_natural tests -------------------------------------------------------

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
        assert_eq!(BaccShoe::no_natural(&hand(player), &hand(banker)), expected);
    }

    #[test]
    #[should_panic]
    fn no_natural_panics_if_player_has_third() {
        let p = hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        let b = hand(&[CardInt::Card4s, CardInt::Card5h]);
        BaccShoe::no_natural(&p, &b);
    }

    #[test]
    #[should_panic]
    fn no_natural_panics_if_banker_has_third() {
        let p = hand(&[CardInt::Card4s, CardInt::Card5h]);
        let b = hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        BaccShoe::no_natural(&p, &b);
    }

    // -- stand_pat tests --------------------------------------------------------

    #[rstest]
    #[case(&[CardInt::Card3s, CardInt::Card3h], false)] // value=6: stands
    #[case(&[CardInt::Card3s, CardInt::Card4h], false)] // value=7: stands
    #[case(&[CardInt::CardKs, CardInt::Card5h], true)] // value=5: draws
    #[case(&[CardInt::Card2s, CardInt::Card3h], true)] // value=5: draws
    #[case(&[CardInt::CardKs, CardInt::CardKh], true)] // value=0: draws
    fn stand_pat_cases(#[case] cards: &[CardInt], #[case] draws: bool) {
        assert_eq!(BaccShoe::stand_pat(&hand(cards)), !draws);
    }

    #[test]
    #[should_panic]
    fn stand_pat_panics_if_hand_has_third() {
        let h = hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        BaccShoe::stand_pat(&h);
    }

    // -- banker_take_third tests ------------------------------------------------

    // score 0-2: always draws regardless of player third card
    #[rstest]
    #[case(&[CardInt::CardKs, CardInt::CardKh], CardInt::CardAs)] // value=0
    #[case(&[CardInt::CardKs, CardInt::CardAh], CardInt::Card8s)] // value=1
    #[case(&[CardInt::CardKs, CardInt::Card2h], CardInt::Card8s)] // value=2
    fn score_0_to_2_always_draws(#[case] banker: &[CardInt], #[case] player_third: CardInt) {
        assert!(BaccShoe::banker_take_third(&hand(banker), player_third));
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
        let banker = hand(&[CardInt::CardAs, CardInt::Card2h]); // value=3
        assert_eq!(BaccShoe::banker_take_third(&banker, player_third), expected);
    }

    // score 4: draws on pip 2-7
    #[rstest]
    #[case(CardInt::Card2s, true)] // pip=2: boundary low
    #[case(CardInt::Card5s, true)] // pip=5: mid
    #[case(CardInt::Card7s, true)] // pip=7: boundary high
    #[case(CardInt::CardAs, false)] // pip=1: below range
    #[case(CardInt::Card8s, false)] // pip=8: above range
    fn score_4(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = hand(&[CardInt::Card2s, CardInt::Card2h]); // value=4
        assert_eq!(BaccShoe::banker_take_third(&banker, player_third), expected);
    }

    // score 5: draws on pip 4-7
    #[rstest]
    #[case(CardInt::Card4s, true)] // pip=4: boundary low
    #[case(CardInt::Card6s, true)] // pip=6: mid
    #[case(CardInt::Card7s, true)] // pip=7: boundary high
    #[case(CardInt::Card3s, false)] // pip=3: below range
    #[case(CardInt::Card8s, false)] // pip=8: above range
    fn score_5(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = hand(&[CardInt::Card2s, CardInt::Card3h]); // value=5
        assert_eq!(BaccShoe::banker_take_third(&banker, player_third), expected);
    }

    // score 6: draws on pip 6-7 (covered by iterator tests; boundary cases added here)
    #[rstest]
    #[case(CardInt::Card6s, true)] // pip=6: boundary low
    #[case(CardInt::Card7s, true)] // pip=7: boundary high
    #[case(CardInt::Card5s, false)] // pip=5: below range
    #[case(CardInt::Card8s, false)] // pip=8: above range
    fn score_6(#[case] player_third: CardInt, #[case] expected: bool) {
        let banker = hand(&[CardInt::Card3s, CardInt::Card3h]); // value=6
        assert_eq!(BaccShoe::banker_take_third(&banker, player_third), expected);
    }

    // score 7: never draws
    #[rstest]
    #[case(CardInt::CardAs)]
    #[case(CardInt::Card8s)]
    #[case(CardInt::CardKs)]
    fn score_7_never_draws(#[case] player_third: CardInt) {
        let banker = hand(&[CardInt::Card3s, CardInt::Card4h]); // value=7
        assert!(!BaccShoe::banker_take_third(&banker, player_third));
    }

    #[test]
    #[should_panic]
    fn banker_take_third_panics_if_banker_has_third() {
        let banker = hand(&[CardInt::CardAs, CardInt::Card2h, CardInt::Card3d]);
        BaccShoe::banker_take_third(&banker, CardInt::Card4s);
    }

    // -- Iterator trait tests -------------------------------------------------
    //
    // Each test builds a shoe via BaccShoe::from(&[Card]), drives one
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
    //   skips the Cut via deal's loop, deals all play cards, returns Some.
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
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[c3, c1]);
        assert_eq!(round.banker_cards(), &[c2, c0]);
        assert_eq!(round.cut_card_index(), Some(0));
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
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[r4, r2]);
        assert_eq!(round.banker_cards(), &[r3, r1, r0]);
        assert_eq!(round.cut_card_index(), Some(0));
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
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[r4, r2, r0]);
        assert_eq!(round.banker_cards(), &[r3, r1]);
        assert_eq!(round.cut_card_index(), Some(0));
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
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.player_cards(), &[r5, r3, r1]);
        assert_eq!(round.banker_cards(), &[r4, r2, r0]);
        assert_eq!(round.cut_card_index(), Some(0));
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
        //   deal p1 : deals cards[9]=Card9s              cursor->8
        //   deal b1 : deals cards[8]=Card5h              cursor->7
        //   deal p2 : deals cards[7]=Card::Cut -> skip;  cursor->6
        //             deals cards[6]=CardKs              cursor->5  -> p2=CardKs
        //   deal b2 : deals cards[5]=Card2d              cursor->4
        //   player=[Card9s,CardKs] value=9 (natural) -> no thirds.
        //   Returns Some(round1). After return: cursor=4 <= cut_pos=7 -> exhausted on next check.
        //
        // Round 2 next(): has_reached_cut_card()=true (4<=7) -> is_exhausted=true.
        //   deal p1 : cards[4]=Card2h  cursor->3
        //   deal b1 : cards[3]=Card9c  cursor->2
        //   deal p2 : cards[2]=Card5c  cursor->1
        //   deal b2 : cards[1]=CardKd  cursor->0
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
            Card::Cut,                   // cut card - encountered during round 1's p2 deal
            Card::Play(CardInt::Card5h), // round 1 banker card 1
            Card::Play(CardInt::Card9s), // round 1 player card 1
            Card::Play(CardInt::Card3c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));

        let round1 = shoe.next().expect("round 1 should be dealt");
        assert_eq!(round1.player_cards(), &[CardInt::Card9s, CardInt::CardKs]);
        assert_eq!(round1.banker_cards(), &[CardInt::Card5h, CardInt::Card2d]);
        assert_eq!(round1.cut_card_index(), Some(2));

        let round2 = shoe.next().expect("round 2 should be dealt");
        assert_eq!(round2.player_cards(), &[CardInt::Card2h, CardInt::Card5c]);
        assert_eq!(round2.banker_cards(), &[CardInt::Card9c, CardInt::CardKd]);
        assert_eq!(round2.cut_card_index(), None);

        assert!(shoe.next().is_none());
    }

    // -- cut_card_index tests ------------------------------------------------

    #[test]
    fn cut_card_index_at_0() {
        // Cut immediately after burn, before player card 1.
        // has_reached_cut_card() is true at the start of next() -> this IS the last round.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::CardKd), // banker card 2
            Card::Play(CardInt::CardKh), // player card 2
            Card::Play(CardInt::CardKs), // banker card 1
            Card::Play(CardInt::Card9s), // player card 1 (natural)
            Card::Cut,
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.cut_card_index(), Some(0));
        assert!(shoe.next().is_none());
    }

    #[test]
    fn cut_card_index_at_1() {
        // Cut before banker card 1.
        // Round 2 cards (indices 1-4) give the last round after this one.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card2d), // round 2 banker card 2
            Card::Play(CardInt::CardAh), // round 2 player card 2
            Card::Play(CardInt::Card4d), // round 2 banker card 1
            Card::Play(CardInt::Card8h), // round 2 player card 1 (natural: 8+A=9)
            Card::Play(CardInt::CardKd), // banker card 2
            Card::Play(CardInt::CardKh), // player card 2
            Card::Play(CardInt::CardKs), // banker card 1
            Card::Cut,
            Card::Play(CardInt::Card9s), // player card 1 (natural)
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.cut_card_index(), Some(1));
        let last = shoe
            .next()
            .expect("one more round follows when cut card index > 0");
        assert_eq!(last.cut_card_index(), None);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn cut_card_index_at_2() {
        // Cut before player card 2.
        // Round 2 cards (indices 1-4) give the last round after this one.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card2d), // round 2 banker card 2
            Card::Play(CardInt::CardAh), // round 2 player card 2
            Card::Play(CardInt::Card4d), // round 2 banker card 1
            Card::Play(CardInt::Card8h), // round 2 player card 1 (natural: 8+A=9)
            Card::Play(CardInt::CardKd), // banker card 2
            Card::Play(CardInt::CardKh), // player card 2
            Card::Cut,
            Card::Play(CardInt::CardKs), // banker card 1
            Card::Play(CardInt::Card9s), // player card 1 (natural)
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.cut_card_index(), Some(2));
        let last = shoe
            .next()
            .expect("one more round follows when cut card index > 0");
        assert_eq!(last.cut_card_index(), None);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn cut_card_index_at_3() {
        // Cut before banker card 2.
        // Round 2 cards (indices 1-4) give the last round after this one.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card2d), // round 2 banker card 2
            Card::Play(CardInt::CardAh), // round 2 player card 2
            Card::Play(CardInt::Card4d), // round 2 banker card 1
            Card::Play(CardInt::Card8h), // round 2 player card 1 (natural: 8+A=9)
            Card::Play(CardInt::CardKd), // banker card 2
            Card::Cut,
            Card::Play(CardInt::CardKh), // player card 2
            Card::Play(CardInt::CardKs), // banker card 1
            Card::Play(CardInt::Card9s), // player card 1 (natural)
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.cut_card_index(), Some(3));
        let last = shoe
            .next()
            .expect("one more round follows when cut card index > 0");
        assert_eq!(last.cut_card_index(), None);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn cut_card_index_at_4() {
        // player=[2s,3h] value=5 -> draws; banker=[3s,4h] value=7 -> stands. Cut before player card 3.
        // Round 2 cards (indices 1-4) give the last round after this one.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card2d), // round 2 banker card 2
            Card::Play(CardInt::CardAh), // round 2 player card 2
            Card::Play(CardInt::Card4d), // round 2 banker card 1
            Card::Play(CardInt::Card8h), // round 2 player card 1 (natural: 8+A=9)
            Card::Play(CardInt::Card5c), // player card 3
            Card::Cut,
            Card::Play(CardInt::Card4h), // banker card 2
            Card::Play(CardInt::Card3h), // player card 2
            Card::Play(CardInt::Card3s), // banker card 1
            Card::Play(CardInt::Card2s), // player card 1
            Card::Play(CardInt::CardKc), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.cut_card_index(), Some(4));
        let last = shoe
            .next()
            .expect("one more round follows when cut card index > 0");
        assert_eq!(last.cut_card_index(), None);
        assert!(shoe.next().is_none());
    }

    #[test]
    fn cut_card_index_at_5() {
        // player=[2s,3h] value=5 -> draws p3=6c (pip=6); banker=[3s,3d] value=6 -> draws. Cut before banker card 3.
        // Round 2 cards (indices 1-4) give the last round after this one.
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card2d), // round 2 banker card 2
            Card::Play(CardInt::CardAh), // round 2 player card 2
            Card::Play(CardInt::Card4d), // round 2 banker card 1
            Card::Play(CardInt::Card8h), // round 2 player card 1 (natural: 8+A=9)
            Card::Play(CardInt::CardKs), // banker card 3
            Card::Cut,
            Card::Play(CardInt::Card6c), // player card 3 (pip=6)
            Card::Play(CardInt::Card3d), // banker card 2
            Card::Play(CardInt::Card3h), // player card 2
            Card::Play(CardInt::Card3s), // banker card 1
            Card::Play(CardInt::Card2s), // player card 1
            Card::Play(CardInt::CardKc), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.cut_card_index(), Some(5));
        let last = shoe
            .next()
            .expect("one more round follows when cut card index > 0");
        assert_eq!(last.cut_card_index(), None);
        assert!(shoe.next().is_none());
    }

    // -- banker_forced_third flag tests --------------------------------------

    #[rstest]
    // banker score 0 (K+K=0): always draws -> forced
    #[case(CardInt::CardKs, CardInt::CardKh, CardInt::Card4s, true)]
    // banker score 1 (K+A=1): always draws -> forced
    #[case(CardInt::CardKs, CardInt::CardAh, CardInt::Card4s, true)]
    // banker score 2 (K+2=2): always draws -> forced (upper boundary)
    #[case(CardInt::CardKs, CardInt::Card2h, CardInt::Card4s, true)]
    // banker score 3 (A+2=3), player pip=7 (!=8): banker draws but score above forced threshold
    #[case(CardInt::CardAs, CardInt::Card2h, CardInt::Card7s, false)]
    // banker score 3 (A+2=3), player pip=8: banker does not draw at all
    #[case(CardInt::CardAs, CardInt::Card2h, CardInt::Card8s, false)]
    fn banker_forced_third_player_drew(
        #[case] banker_c1: CardInt,
        #[case] banker_c2: CardInt,
        #[case] player_third: CardInt,
        #[case] expected: bool,
    ) {
        // player=[2s,3h] value=5 -> draws; banker value varies per case.
        // Layout (L=10, dealt right-to-left):
        //   [dummy | b_c3  | p_c3        | b_c2      | p_c2     | b_c1      | p_c1     | Cut | burn  | As ]
        //    idx 0    1       2             3            4          5            6          7     8       9
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card4d), // banker card 3 (placeholder; dealt only if banker draws)
            Card::Play(player_third),    // player card 3
            Card::Play(banker_c2),       // banker card 2
            Card::Play(CardInt::Card3h), // player card 2
            Card::Play(banker_c1),       // banker card 1
            Card::Play(CardInt::Card2s), // player card 1
            Card::Cut,
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert_eq!(round.is_forced_third(), expected);
        assert_eq!(round.cut_card_index(), Some(0));
    }

    #[test]
    fn banker_forced_third_false_when_player_stands() {
        // player=[3s,3h] value=6 -> stand_pat; banker=[Ks,Kh] value=0 -> draws in the
        // independent branch. banker_forced_third must remain false because the flag
        // is only set inside the player-drew branch.
        // Layout (L=9, dealt right-to-left):
        //   [dummy | b_c3  | b_c2 | p_c2 | b_c1 | p_c1 | Cut | burn | As ]
        //    idx 0    1       2      3      4      5       6     7      8
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::CardAd), // banker card 3
            Card::Play(CardInt::CardKh), // banker card 2
            Card::Play(CardInt::Card3h), // player card 2
            Card::Play(CardInt::CardKs), // banker card 1
            Card::Play(CardInt::Card3s), // player card 1
            Card::Cut,
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert!(!round.is_forced_third());
        assert_eq!(round.cut_card_index(), Some(0));
    }

    #[test]
    fn banker_forced_third_false_on_natural() {
        // player=[9s,Kh] value=9 -> natural; no third cards drawn at all.
        // Layout (L=8, dealt right-to-left):
        //   [dummy | b_c2 | p_c2 | b_c1 | p_c1 | Cut | burn | As ]
        //    idx 0    1      2      3      4      5     6      7
        let cards = vec![
            Card::Play(CardInt::CardJc), // dummy - never dealt (cards[0])
            Card::Play(CardInt::Card2d), // banker card 2
            Card::Play(CardInt::CardKh), // player card 2
            Card::Play(CardInt::Card5h), // banker card 1
            Card::Play(CardInt::Card9s), // player card 1 (natural)
            Card::Cut,
            Card::Play(CardInt::Card5c), // burn card
            Card::Play(CardInt::CardAs), // first card (pip=1, burns 1)
        ];
        let mut shoe = BaccShoe::from(Shoe::from(cards.as_slice()));
        let round = shoe.next().expect("round should be dealt");
        assert!(!round.is_forced_third());
        assert_eq!(round.cut_card_index(), Some(0));
    }
}
