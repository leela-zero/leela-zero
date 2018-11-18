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

#include "FastBoard.h"

FastBoard create_filled_3x3() {
    FastBoard b;
    b.reset_board(3);
    b.set_state(1, 1, FastBoard::BLACK);
    b.set_state(2, 1, FastBoard::BLACK);
    b.set_state(0, 1, FastBoard::WHITE);
    b.set_state(1, 0, FastBoard::WHITE);
    b.set_state(2, 2, FastBoard::BLACK);
    return b;
}

FastBoard create_filled_5x5() {
    FastBoard b;
    b.reset_board(5);
    b.set_state(1, 1, FastBoard::BLACK);
    b.set_state(2, 1, FastBoard::BLACK);
    b.set_state(3, 1, FastBoard::WHITE);
    b.set_state(2, 2, FastBoard::WHITE);
    b.set_state(3, 2, FastBoard::BLACK);
    b.set_state(0, 3, FastBoard::BLACK);
    b.set_state(2, 3, FastBoard::WHITE);
    b.set_state(2, 4, FastBoard::WHITE);
    return b;
}

FastBoard create_filled_9x9() {
    FastBoard b;
    b.reset_board(9);
    b.set_state(5, 4, FastBoard::WHITE);
    b.set_state(5, 3, FastBoard::BLACK);
    b.set_state(4, 5, FastBoard::WHITE);
    b.set_state(2, 2, FastBoard::BLACK);
    b.set_state(4, 3, FastBoard::WHITE);
    b.set_state(1, 2, FastBoard::BLACK);
    b.set_state(6, 3, FastBoard::WHITE);
    b.set_state(2, 3, FastBoard::BLACK);
    b.set_state(5, 2, FastBoard::WHITE);
    b.set_state(0, 0, FastBoard::BLACK);
    b.set_state(6, 6, FastBoard::WHITE);
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
FastBoard create_5x5_all_white_field() {
    FastBoard b;
    b.reset_board(5);
    b.set_state(1, 2, FastBoard::WHITE);
    b.set_state(2, 1, FastBoard::WHITE);
    b.set_state(2, 2, FastBoard::WHITE);
    b.set_state(2, 3, FastBoard::WHITE);
    b.set_state(2, 4, FastBoard::WHITE);
    b.set_state(3, 2, FastBoard::WHITE);
    b.set_state(3, 4, FastBoard::WHITE);
    b.set_state(4, 3, FastBoard::WHITE);
    b.set_state(0, 2, FastBoard::WHITE);
    b.set_state(2, 0, FastBoard::WHITE);
    return b;
}


TEST(FastBoardTest, Board3x3) {
    FastBoard b;
    b.reset_board(3);
    const char *expected = "\n"
                           "   a b c \n"
                           " 3 . . .  3\n"
                           " 2 . . .  2\n"
                           " 1 . . .  1\n"
                           "   a b c \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
    EXPECT_EQ(3, b.get_boardsize());
}

TEST(FastBoardTest, MakeBlackMoveOn19x19) {
    FastBoard b;
    b.reset_board(19);
    b.set_state(b.get_vertex(2, 1), FastBoard::BLACK);

    const char *expected = "\n"
                           "   a b c d e f g h j k l m n o p q r s t \n"
                           "19 . . . . . . . . . . . . . . . . . . . 19\n"
                           "18 . . . . . . . . . . . . . . . . . . . 18\n"
                           "17 . . . . . . . . . . . . . . . . . . . 17\n"
                           "16 . . . + . . . . . + . . . . . + . . . 16\n"
                           "15 . . . . . . . . . . . . . . . . . . . 15\n"
                           "14 . . . . . . . . . . . . . . . . . . . 14\n"
                           "13 . . . . . . . . . . . . . . . . . . . 13\n"
                           "12 . . . . . . . . . . . . . . . . . . . 12\n"
                           "11 . . . . . . . . . . . . . . . . . . . 11\n"
                           "10 . . . + . . . . . + . . . . . + . . . 10\n"
                           " 9 . . . . . . . . . . . . . . . . . . .  9\n"
                           " 8 . . . . . . . . . . . . . . . . . . .  8\n"
                           " 7 . . . . . . . . . . . . . . . . . . .  7\n"
                           " 6 . . . . . . . . . . . . . . . . . . .  6\n"
                           " 5 . . . . . . . . . . . . . . . . . . .  5\n"
                           " 4 . . . + . . . . . + . . . . . + . . .  4\n"
                           " 3 . . . . . . . . . . . . . . . . . . .  3\n"
                           " 2 . . X . . . . . . . . . . . . . . . .  2\n"
                           " 1 . . . . . . . . . . . . . . . . . . .  1\n"
                           "   a b c d e f g h j k l m n o p q r s t \n\n";
    EXPECT_EQ(expected, b.serialize_board());
}

TEST(FastBoardTest, GetVertexOn19x19) {
    FastBoard b;
    b.reset_board(19);
    EXPECT_EQ(22, b.get_vertex(0, 0));
    EXPECT_EQ(43, b.get_vertex(0, 1));
    EXPECT_EQ(44, b.get_vertex(1, 1));
    EXPECT_EQ(87, b.get_vertex(2, 3));
    EXPECT_EQ(418, b.get_vertex(18, 18));
}

TEST(FastBoardTest, GetXYFromVertex) {
    FastBoard b;
    b.reset_board(19);
    EXPECT_EQ(std::make_pair(0, 0), b.get_xy(22));
    EXPECT_EQ(std::make_pair(0, 1), b.get_xy(43));
    EXPECT_EQ(std::make_pair(1, 1), b.get_xy(44));
    EXPECT_EQ(std::make_pair(2, 1), b.get_xy(45));
    EXPECT_EQ(std::make_pair(2, 3), b.get_xy(87));
    EXPECT_EQ(std::make_pair(18, 18), b.get_xy(418));

    // Negative test to check assertion
    // Commenting out until assertions are enable in CI build.
    //ASSERT_DEATH({ b.get_xy(7);}, ".*FastBoard.cpp.* Assertion `y >= 0 && y < m_boardsize' failed.");
}

TEST(FastBoardTest, GetSquare) {
    FastBoard b;
    b.reset_board(19);
    EXPECT_EQ(FastBoard::EMPTY, b.get_state(43));
    EXPECT_EQ(FastBoard::EMPTY, b.get_state(0, 1));
    b.set_state(43, FastBoard::BLACK);
    EXPECT_EQ(FastBoard::BLACK, b.get_state(43));
    b.set_state(43, FastBoard::WHITE);
    EXPECT_EQ(FastBoard::WHITE, b.get_state(43));
}

TEST(FastBoardTest, SemiFilled5x5Board) {
    FastBoard b = create_filled_5x5();

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

// Results will make more sense in FullBuard test
TEST(FastBoardTest, CountRealLibertiesOn5x5) {
    FastBoard b = create_filled_5x5();
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(0, 0)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(1, 1)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(2, 1)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(3, 1)));
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(4, 1)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(2, 2)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(3, 2)));
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(0, 3)));
}

TEST(FastBoardTest, SemiFilled9x9Board) {
    FastBoard b = create_filled_9x9();

    const char *expected = "\n"
                           "   a b c d e f g h j \n"
                           " 9 . . . . . . . . .  9\n"
                           " 8 . . . . . . . . .  8\n"
                           " 7 . . + . + . O . .  7\n"
                           " 6 . . . . O . . . .  6\n"
                           " 5 . . + . + O + . .  5\n"
                           " 4 . . X . O X O . .  4\n"
                           " 3 . X X . + O + . .  3\n"
                           " 2 . . . . . . . . .  2\n"
                           " 1 X . . . . . . . .  1\n"
                           "   a b c d e f g h j \n\n";

    EXPECT_EQ(expected,  b.serialize_board());
}

// Results will make more sense in FullBuard test
TEST(FastBoardTest, CountRealLibertiesOn9x9) {
    FastBoard b = create_filled_9x9();

    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(0, 0)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(1, 2)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(4, 3)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(4, 4)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(5, 4)));
}

// Results will make more sense in FullBuard test
TEST(FastBoardTest, IsSuicideWhenNotForBlack) {
    FastBoard b;
    b.reset_board(5);
    b.set_state(2, 2, FastBoard::WHITE);
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(1, 1), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(2, 1), FastBoard::BLACK));
}

// Results will make more sense in FullBuard test
TEST(FastBoardTest, IsSuicideForBlackInAllWhiteField) {
    FastBoard b = create_5x5_all_white_field();

    EXPECT_EQ(false, b.is_suicide(b.get_vertex(1, 1), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(3, 3), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 4), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 2), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 4), FastBoard::BLACK));
}

TEST(FastBoardTest, CalcAreaScore) {
    FastBoard b = create_filled_5x5();
    EXPECT_EQ(-6.5, b.area_score(6.5F));
    EXPECT_EQ(-.5, b.area_score(0.5F));
    EXPECT_EQ(-9.0, b.area_score(9.0F));
}

TEST(FastBoardTest, CalcAreaScoreOnWhiteField) {
    FastBoard b = create_5x5_all_white_field();
    EXPECT_EQ(-31.5, b.area_score(6.5F));
    EXPECT_EQ(-25.5, b.area_score(0.5F));
    EXPECT_EQ(-34.0, b.area_score(9.0F));
}

TEST(FastBoardTest, CalcAreaScoreOnSemiFilled9x9) {
    FastBoard b = create_filled_9x9();
    EXPECT_EQ(-7.5, b.area_score(6.5F));
    EXPECT_EQ(-1.5, b.area_score(0.5F));
    EXPECT_EQ(-10.0, b.area_score(9.0F));
}

TEST(FastBoardTest, ToMove) {
    FastBoard b = create_filled_5x5();
    EXPECT_EQ(FastBoard::BLACK, b.get_to_move());
    EXPECT_EQ(true, b.black_to_move());
    b.set_to_move(FastBoard::WHITE);
    EXPECT_EQ(FastBoard::WHITE, b.get_to_move());
    EXPECT_EQ(false, b.black_to_move());
}

TEST(FastBoardTest, MoveToText) {
    FastBoard b = create_filled_3x3();
    EXPECT_EQ("B1", b.move_to_text(b.get_vertex(1, 0)));
    EXPECT_EQ("A2", b.move_to_text(b.get_vertex(0, 1)));
    EXPECT_EQ("pass", b.move_to_text(FastBoard::PASS));
    EXPECT_EQ("resign", b.move_to_text(FastBoard::RESIGN));
}

TEST(FastBoardTest, MoveToTextSgf) {
    FastBoard b = create_filled_3x3();
    EXPECT_EQ("bc", b.move_to_text_sgf(b.get_vertex(1, 0)));
    EXPECT_EQ("ab", b.move_to_text_sgf(b.get_vertex(0, 1)));
    EXPECT_EQ("ca", b.move_to_text_sgf(b.get_vertex(2, 2)));
    EXPECT_EQ("tt", b.move_to_text_sgf(FastBoard::PASS));
    EXPECT_EQ("tt", b.move_to_text_sgf(FastBoard::RESIGN));
}

TEST(FastBoardTest, GetStoneList) {
    FastBoard emptyBoard;
    emptyBoard.reset_board(3);
    EXPECT_EQ("", emptyBoard.get_stone_list());

    FastBoard b = create_filled_5x5();
    EXPECT_EQ("A4 B2 C2 C3 C4 C5 D2 D3", b.get_stone_list());

    FastBoard whiteFieldBoard = create_5x5_all_white_field();
    EXPECT_EQ("A3 B3 C1 C2 C3 C4 C5 D3 D5 E4", whiteFieldBoard.get_stone_list());
}

TEST(FastBoardTest, StarPoint9x9) {

    EXPECT_EQ(true, FastBoard::starpoint(9, 2, 2));
    EXPECT_EQ(true, FastBoard::starpoint(9, 4, 4));
    EXPECT_EQ(false, FastBoard::starpoint(9, 5, 5));
    EXPECT_EQ(false, FastBoard::starpoint(9, 3, 4));
}

TEST(FastBoardTest, StarPoint13x13) {

    EXPECT_EQ(false, FastBoard::starpoint(13, 2, 2));
    EXPECT_EQ(true, FastBoard::starpoint(13, 3, 3));
    EXPECT_EQ(false, FastBoard::starpoint(13, 4, 4));
    EXPECT_EQ(true, FastBoard::starpoint(13, 6, 6));
    EXPECT_EQ(false, FastBoard::starpoint(13, 2, 3));
    EXPECT_EQ(false, FastBoard::starpoint(13, 8, 8));
}

TEST(FastBoardTest, StarPoint19x19) {

    EXPECT_EQ(false, FastBoard::starpoint(19, 2, 2));
    EXPECT_EQ(false, FastBoard::starpoint(19, 4, 4));
    EXPECT_EQ(false, FastBoard::starpoint(19, 2, 3));
    EXPECT_EQ(true, FastBoard::starpoint(19, 3, 3));
    EXPECT_EQ(true, FastBoard::starpoint(19, 15, 15));
    EXPECT_EQ(false, FastBoard::starpoint(19, 14, 14));
    EXPECT_EQ(false, FastBoard::starpoint(19, 3, 14));
    EXPECT_EQ(true, FastBoard::starpoint(19, 3, 15));
    EXPECT_EQ(true, FastBoard::starpoint(19, 3, 9));
}