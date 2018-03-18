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

#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "FastBoard.h"


FastBoard createSemiFilled5x5() {
    FastBoard b;
    b.reset_board(5);
    b.set_square(1, 1, FastBoard::BLACK);
    b.set_square(2, 1, FastBoard::BLACK);
    b.set_square(3, 1, FastBoard::WHITE);
    b.set_square(2, 2, FastBoard::WHITE);
    b.set_square(3, 2, FastBoard::BLACK);
    b.set_square(0, 3, FastBoard::BLACK);
    b.set_square(2, 3, FastBoard::WHITE);
    b.set_square(2, 4, FastBoard::WHITE);
    return b;
}


FastBoard createSemiFilled9x9() {
    FastBoard b;
    b.reset_board(9);
    b.set_square(5, 4, FastBoard::WHITE);
    b.set_square(5, 3, FastBoard::BLACK);
    b.set_square(4, 5, FastBoard::WHITE);
    b.set_square(2, 2, FastBoard::BLACK);
    b.set_square(4, 3, FastBoard::WHITE);
    b.set_square(1, 2, FastBoard::BLACK);
    b.set_square(6, 3, FastBoard::WHITE);
    b.set_square(2, 3, FastBoard::BLACK);
    b.set_square(5, 2, FastBoard::WHITE);
    b.set_square(0, 0, FastBoard::BLACK);
    b.set_square(6, 6, FastBoard::WHITE);
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
FastBoard create5x5AllWhiteField() {
    FastBoard b;
    b.reset_board(5);
    b.set_square(1, 2, FastBoard::WHITE);
    b.set_square(2, 1, FastBoard::WHITE);
    b.set_square(2, 2, FastBoard::WHITE);
    b.set_square(2, 3, FastBoard::WHITE);
    b.set_square(2, 4, FastBoard::WHITE);
    b.set_square(3, 2, FastBoard::WHITE);
    b.set_square(3, 4, FastBoard::WHITE);
    b.set_square(4, 3, FastBoard::WHITE);
    b.set_square(0, 2, FastBoard::WHITE);
    b.set_square(2, 0, FastBoard::WHITE);
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
    b.set_square(b.get_vertex(2, 1), FastBoard::BLACK);
    
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
    
    EXPECT_EQ(std::make_pair(6, -1), b.get_xy(7)); // should fail
    //ASSERT_DEATH({ b.get_xy(7);}, "failed assertion");
}

TEST(FastBoardTest, GetSquare) {
    FastBoard b;
    b.reset_board(19);
    EXPECT_EQ(FastBoard::EMPTY, b.get_square(43));
    EXPECT_EQ(FastBoard::EMPTY, b.get_square(0, 1));
    b.set_square(43, FastBoard::BLACK);
    EXPECT_EQ(FastBoard::BLACK, b.get_square(43));
    b.set_square(43, FastBoard::WHITE);
    EXPECT_EQ(FastBoard::WHITE, b.get_square(43));
}

TEST(FastBoardTest, SemiFilled5x5Board) {
    FastBoard b = createSemiFilled5x5();
    
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

TEST(FastBoardTest, CountRealLibertiesOn5x5) {
    FastBoard b = createSemiFilled5x5();
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(0, 0)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(1, 1)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(2, 1)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(3, 1))); // ?
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(4, 1))); // ?
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(2, 2))); // ? 
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(3, 2))); // ?
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(0, 3)));
}

TEST(FastBoardTest, SemiFilled9x9Board) {
    FastBoard b = createSemiFilled9x9();
    
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

TEST(FastBoardTest, CountRealLibertiesOn9x9) {
    FastBoard b = createSemiFilled5x5();
    
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(0, 0)));
    EXPECT_EQ(4, b.count_pliberties(b.get_vertex(1, 2)));
    EXPECT_EQ(3, b.count_pliberties(b.get_vertex(4, 3)));
    EXPECT_EQ(2, b.count_pliberties(b.get_vertex(4, 4))); 
    EXPECT_EQ(0, b.count_pliberties(b.get_vertex(5, 4)));
}

TEST(FastBoardTest, IsSuicideWhenNotForBlack) {
    FastBoard b;
    b.reset_board(5);  
    b.set_square(2, 2, FastBoard::WHITE);
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(1, 1), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(2, 1), FastBoard::BLACK));
}

TEST(FastBoardTest, IsSuicideWhenForBlackInAllWhiteField) {
    FastBoard b = create5x5AllWhiteField();

    EXPECT_EQ(false, b.is_suicide(b.get_vertex(1, 1), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(3, 3), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 4), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 2), FastBoard::BLACK));
    EXPECT_EQ(false, b.is_suicide(b.get_vertex(4, 4), FastBoard::BLACK));
}
