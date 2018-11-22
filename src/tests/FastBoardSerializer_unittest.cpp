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

#include "FastBoardSerializer.h"



TEST(FastBoardSerializerTest, StarPoint9x9) {

    EXPECT_EQ(true, FastBoardSerializer::starpoint(9, 2, 2));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(9, 4, 4));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(9, 5, 5));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(9, 3, 4));
}

TEST(FastBoardSerializerTest, StarPoint13x13) {

    EXPECT_EQ(false, FastBoardSerializer::starpoint(13, 2, 2));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(13, 3, 3));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(13, 4, 4));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(13, 6, 6));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(13, 2, 3));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(13, 8, 8));
}

TEST(FastBoardSerializerTest, StarPoint19x19) {

    EXPECT_EQ(false, FastBoardSerializer::starpoint(19, 2, 2));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(19, 4, 4));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(19, 2, 3));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(19, 3, 3));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(19, 15, 15));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(19, 14, 14));
    EXPECT_EQ(false, FastBoardSerializer::starpoint(19, 3, 14));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(19, 3, 15));
    EXPECT_EQ(true, FastBoardSerializer::starpoint(19, 3, 9));
}
