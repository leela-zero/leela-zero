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


TEST(FastBoardTest, Board3x3) {
    FastBoard b;
    b.reset_board(3);
    EXPECT_EQ(
        "\n   a b c \n 3 . . .  3\n 2 . . .  2\n 1 . . .  1\n   a b c \n\n",  
        b.serialize_board()
    );
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
