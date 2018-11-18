/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#undef NDEBUG
#include <cassert>
#include <cstddef>
#include <gtest/gtest.h>

#include "FullBoard.h"

FullBoard create_full_filled_3x3() {
    FullBoard b;
    b.reset_board(3);
    b.update_board(FastBoard::BLACK, b.get_vertex(1, 1));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(0, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(1, 0));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 2));
    return b;
}

/*
      a b c d e
    5 . . O . .  5
    4 X . O . .  4
    3 . . O X .  3
    2 . X X O .  2
    1 . . . . .  1
      a b c d e
*/
FullBoard create_semi_filled_5x5() {
    FullBoard b;
    b.reset_board(5);
    b.update_board(FastBoard::BLACK, b.get_vertex(1, 1));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(3, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(3, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(0, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 4));
    return b;
}

/**
       a b c d e
     5 . . O . O  5
     4 X . O O X  4
     3 X O O X .  3
     2 . X X O .  2
     1 O X . X .  1
       a b c d e
 */
FullBoard create_full_filled_5x5() {
    FullBoard b;
    b.reset_board(5);
    b.update_board(FastBoard::BLACK, b.get_vertex(1, 1));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(3, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(3, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(0, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 4));
    b.update_board(FastBoard::WHITE, b.get_vertex(3, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(4, 4));
    b.update_board(FastBoard::BLACK, b.get_vertex(4, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(1, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(0, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(0, 0));
    b.update_board(FastBoard::BLACK, b.get_vertex(1, 0));
    b.update_board(FastBoard::BLACK, b.get_vertex(3, 0));
    return b;
}

/*
         a b c d e
       5 X . X X .  5
       4 . . X . X  4
       3 X X . X .  3
       2 . O X O .  2
       1 . . . . .  1
         a b c d e
*/
FullBoard create_5x5_all_black() {
    FullBoard b;
    b.reset_board(5);
    b.update_board(FastBoard::BLACK, b.get_vertex(1, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 1));
    b.update_board(FastBoard::BLACK, b.get_vertex(0, 4));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 3));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 4));
    b.update_board(FastBoard::BLACK, b.get_vertex(3, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(3, 4));
    b.update_board(FastBoard::BLACK, b.get_vertex(4, 3));
    b.update_board(FastBoard::BLACK, b.get_vertex(0, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(1, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(3, 1));
    return b;
}

FullBoard create_semi_filled_9x9() {
    FullBoard b;
    b.reset_board(9);
    b.update_board(FastBoard::WHITE, b.get_vertex(5, 4));
    b.update_board(FastBoard::BLACK, b.get_vertex(5, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(4, 5));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(4, 3));
    b.update_board(FastBoard::BLACK, b.get_vertex(1, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(6, 3));
    b.update_board(FastBoard::BLACK, b.get_vertex(2, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(5, 2));
    b.update_board(FastBoard::BLACK, b.get_vertex(0, 0));
    b.update_board(FastBoard::WHITE, b.get_vertex(6, 6));
    return b;
}

/*
      a b c d e
    5 . . O O .  5
    4 . . O . O  4
    3 O O O O .  3
    2 . . O . .  2
    1 . . O . .  1
      a b c d e
*/
FullBoard create_5x5_all_white() {
    FullBoard b;
    b.reset_board(5);
    b.update_board(FastBoard::WHITE, b.get_vertex(1, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 1));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 4));
    b.update_board(FastBoard::WHITE, b.get_vertex(3, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(3, 4));
    b.update_board(FastBoard::WHITE, b.get_vertex(4, 3));
    b.update_board(FastBoard::WHITE, b.get_vertex(0, 2));
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 0));
    return b;
}


TEST(FullBoardTest, Board3x3) {
    FullBoard b;
    b.reset_board(3);
    const char *expected = "\n"
                           "   a b c \n"
                           " 3 . . .  3\n"
                           " 2 . . .  2\n"
                           " 1 . . .  1\n"
                           "   a b c \n\n";

    EXPECT_EQ(expected, b.serialize_board());
    EXPECT_EQ(3, b.get_boardsize());
}

TEST(FullBoardTest, SerializeSemiFilled5x5Board) {
    FullBoard b = create_semi_filled_5x5();

    const char *expected = "\n"
                           "   a b c d e \n"
                           " 5 . . O . .  5\n"
                           " 4 X . O . .  4\n"
                           " 3 . . O X .  3\n"
                           " 2 . X X O .  2\n"
                           " 1 . . . . .  1\n"
                           "   a b c d e \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
}

TEST(FullBoardTest, SerializeFilled5x5Board) {
    FullBoard b = create_full_filled_5x5();

    const char *expected = "\n"
                           "   a b c d e \n"
                           " 5 . . O . O  5\n"
                           " 4 X . O O X  4\n"
                           " 3 X O O X .  3\n"
                           " 2 . X X O .  2\n"
                           " 1 O X . X .  1\n"
                           "   a b c d e \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
}

TEST(FullBoardTest, SerializeAllBlack5x5Board) {
    FullBoard b = create_5x5_all_black();

    const char *expected = "\n"
                           "   a b c d e \n"
                           " 5 X . X X .  5\n"
                           " 4 . . X . X  4\n"
                           " 3 X X . X .  3\n"
                           " 2 . O X O .  2\n"
                           " 1 . . . . .  1\n"
                           "   a b c d e \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
}


TEST(FullBoardTest, CountRealLibertiesOn5x5) {
    FullBoard b = create_semi_filled_5x5();
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(0, 0)));
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(1, 1)));
    EXPECT_EQ(1, b.count_pliberties(b.get_vertex(2, 1)));
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(3, 1)));
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(4, 1)));
    EXPECT_EQ(1, b.count_pliberties(b.get_vertex(2, 2)));
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(3, 2)));
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(0, 3)));
}

TEST(FullBoardTest, SemiFilled9x9Board) {
    FullBoard b = create_semi_filled_9x9();

    const char *expected = "\n"
                           "   a b c d e f g h j \n"
                           " 9 . . . . . . . . .  9\n"
                           " 8 . . . . . . . . .  8\n"
                           " 7 . . + . + . O . .  7\n"
                           " 6 . . . . O . . . .  6\n"
                           " 5 . . + . + O + . .  5\n"
                           " 4 . . X . O . O . .  4\n"
                           " 3 . X X . + O + . .  3\n"
                           " 2 . . . . . . . . .  2\n"
                           " 1 X . . . . . . . .  1\n"
                           "   a b c d e f g h j \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
}

TEST(FullBoardTest, RemoveString) {
    FullBoard b = create_semi_filled_9x9();
    b.remove_string(b.get_vertex(1, 2));

    const char *expected = "\n"
                           "   a b c d e f g h j \n"
                           " 9 . . . . . . . . .  9\n"
                           " 8 . . . . . . . . .  8\n"
                           " 7 . . + . + . O . .  7\n"
                           " 6 . . . . O . . . .  6\n"
                           " 5 . . + . + O + . .  5\n"
                           " 4 . . . . O . O . .  4\n"
                           " 3 . . + . + O + . .  3\n"
                           " 2 . . . . . . . . .  2\n"
                           " 1 X . . . . . . . .  1\n"
                           "   a b c d e f g h j \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
}

TEST(FullBoardTest, CountRealLibertiesOn9x9) {
    FullBoard b = create_semi_filled_9x9();

    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(0, 0)));
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(1, 2)));
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(2, 2)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(4, 3)));
    EXPECT_EQ(1, b.count_pliberties(b.get_vertex(4, 4)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(5, 4)));
}

TEST(FullBoardTest, IsSuicideWhenNotForBlack) {
    FullBoard b;
    b.reset_board(5);
    b.update_board(FastBoard::WHITE, b.get_vertex(2, 2));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(1, 1), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(2, 1), FastBoard::BLACK));
}

TEST(FullBoardTest, IsSuicideForBlackInAllWhiteField) {
    FullBoard b = create_5x5_all_white();

    EXPECT_EQ(false, b.is_suicide(b.get_vertex(1, 1), FastBoard::BLACK));
    EXPECT_EQ(true, b.is_suicide(b.get_vertex(3, 3), FastBoard::BLACK));
    EXPECT_EQ(true, b.is_suicide(b.get_vertex(4, 4), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 2), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(3, 4), FastBoard::BLACK));
}

TEST(FullBoardTest, CalcAreaScore) {
    FullBoard b = create_semi_filled_5x5();
    EXPECT_EQ(-6.5, b.area_score(6.5F));
    EXPECT_EQ(-.5, b.area_score(0.5F));
    EXPECT_EQ(-9.0, b.area_score(9.0F));
}

TEST(FullBoardTest, CalcAreaScoreOnWhiteField) {
    FullBoard b = create_5x5_all_white();
    EXPECT_EQ(-31.5, b.area_score(6.5F));
    EXPECT_EQ(-25.5, b.area_score(0.5F));
    EXPECT_EQ(-34.0, b.area_score(9.0F));
}

TEST(FullBoardTest, CalcAreaScoreOnSemiFilled9x9) {
    FullBoard b = create_semi_filled_9x9();
    EXPECT_EQ(-9.5, b.area_score(6.5F));
    EXPECT_EQ(-3.5, b.area_score(0.5F));
    EXPECT_EQ(-12.0, b.area_score(9.0F));
}

TEST(FullBoardTest, WhiteToMove) {
    FullBoard b = create_semi_filled_5x5();
    EXPECT_EQ(FastBoard::BLACK, b.get_to_move());
    EXPECT_EQ(8162750142217023897u, b.get_hash());

    b.set_to_move(FastBoard::WHITE);
    EXPECT_EQ(FastBoard::WHITE, b.get_to_move());
    EXPECT_EQ(15747471392336042580u, b.get_hash());
}

TEST(FullBoardTest, BlackToMove) {
    FullBoard b = create_semi_filled_5x5();
    b.set_to_move(FastBoard::WHITE);
    EXPECT_EQ(FastBoard::WHITE, b.get_to_move());
    EXPECT_EQ(15747471392336042580u, b.get_hash());

    b.set_to_move(FastBoard::BLACK);
    EXPECT_EQ(FastBoard::BLACK, b.get_to_move());
    EXPECT_EQ(8162750142217023897u, b.get_hash());
}

TEST(FullBoardTest, CalcHash9x9) {
    FullBoard b = create_semi_filled_9x9();
    EXPECT_EQ(10841953875953604838u, b.calc_hash());
    EXPECT_EQ(16342085426476978742u, b.calc_hash(b.get_vertex(2, 3)));
    EXPECT_EQ(10275374004301650050u, b.calc_hash(b.get_vertex(1, 1)));
}

TEST(FullBoardTest, CalcKoHash9x9) {
    FullBoard b = create_semi_filled_9x9();
    EXPECT_EQ(15992831752030735207u, b.calc_ko_hash());
}

TEST(FullBoardTest, GetString) {
    FullBoard b = create_full_filled_5x5();
    EXPECT_EQ("C2 B1 B2", b.get_string(b.get_vertex(1, 1)));
    EXPECT_EQ("C4 B3 D4 C5 C3", b.get_string(b.get_vertex(2, 2)));
}


/*
  Only single points surrounded by own color are counted as eyes.

      a b c d e
    5 . . O O .  5
    4 . . O . O  4
    3 O O O O .  3
    2 . . O . .  2
    1 . . O . .  1
      a b c d e
*/
TEST(FullBoardTest, IsEyeOnWhiteField) {
    FullBoard b = create_5x5_all_white();
    EXPECT_EQ(true, b.is_eye(FastBoard::WHITE, b.get_vertex(4, 4)));
    EXPECT_EQ(true, b.is_eye(FastBoard::WHITE, b.get_vertex(3, 3)));
    EXPECT_EQ(true, b.is_eye(FastBoard::WHITE, b.get_vertex(2, 2))); // not eye because its filled
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(1, 1))); // not a single point eye
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(1, 4))); // not a single point eye
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(2, 0)));
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(4, 2))); // not surrounded on 4 sides
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(3, 3)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(2, 2)));
}

/*
  Only single points surrounded by own color are counted as eyes.

      a b c d e
    5 X . X X .  5
    4 . . X . X  4
    3 X X . X .  3
    2 . O X O .  2
    1 . . . . .  1
      a b c d e
*/
TEST(FullBoardTest, IsEyeOnBlackField) {
    FullBoard b = create_5x5_all_black();
    EXPECT_EQ(true, b.is_eye(FastBoard::BLACK, b.get_vertex(4, 4)));
    EXPECT_EQ(true, b.is_eye(FastBoard::BLACK, b.get_vertex(3, 3))); // black eye
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(3, 3))); // not whte eye
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(2, 2))); // potentially false eye
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(1, 1)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(0, 3)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(1, 3))); // not single point eye
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(4, 2)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(3, 0)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(1, 1)));
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(1, 1)));
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(2, 0)));
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(4, 2)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(1, 4)));
}

/*
       a b c d e
     5 . . O . O  5
     4 X . O O X  4
     3 X O O X .  3
     2 . X X O .  2
     1 O X . X .  1
       a b c d e
*/
TEST(FullBoardTest, IsEyeOnFull5x5) {
    FullBoard b = create_full_filled_5x5();
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(3, 4))); // false eye
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(4, 2))); // false eye
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(2, 0)));
    EXPECT_EQ(false, b.is_eye(FastBoard::WHITE, b.get_vertex(0, 2)));
    EXPECT_EQ(false, b.is_eye(FastBoard::BLACK, b.get_vertex(3, 1)));
}

TEST(FullBoardTest, GetPrisonersWhenBlackPrisoner) {
    FullBoard b = create_full_filled_5x5();
    EXPECT_EQ(0, b.get_prisoners(FastBoard::WHITE));
    EXPECT_EQ(0, b.get_prisoners(FastBoard::BLACK));

    b.update_board(FastBoard::BLACK, b.get_vertex(3, 4)); // this captures a white stone
    EXPECT_EQ(0, b.get_prisoners(FastBoard::WHITE));
    EXPECT_EQ(1, b.get_prisoners(FastBoard::BLACK));
}

TEST(FullBoardTest, GetPrisonersWhenWhitePrisoner) {
    FullBoard b = create_semi_filled_9x9();
    EXPECT_EQ(1, b.get_prisoners(FastBoard::WHITE));
    EXPECT_EQ(0, b.get_prisoners(FastBoard::BLACK));
}
