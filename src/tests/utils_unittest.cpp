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
#include <gtest/gtest.h>

#include "Utils.h"

using namespace Utils;

TEST(UtilsTest, CeilMultiple) {
    // Equal to a multiple
    EXPECT_EQ(ceilMultiple(0, 1), 0);
    EXPECT_EQ(ceilMultiple(0, 3), 0);

    EXPECT_EQ(ceilMultiple(6,  1), 6);
    EXPECT_EQ(ceilMultiple(23, 1), 23);

    EXPECT_EQ(ceilMultiple(2, 2), 2);
    EXPECT_EQ(ceilMultiple(4, 2), 4);
    EXPECT_EQ(ceilMultiple(6, 2), 6);
    EXPECT_EQ(ceilMultiple(0, 3), 0);
    EXPECT_EQ(ceilMultiple(3, 3), 3);
    EXPECT_EQ(ceilMultiple(9, 3), 9);

    // Requires rounding up
    EXPECT_EQ(ceilMultiple(3, 5), 5);
    EXPECT_EQ(ceilMultiple(6, 5), 10);
    EXPECT_EQ(ceilMultiple(9, 5), 10);
    EXPECT_EQ(ceilMultiple(23, 5), 25);
    EXPECT_EQ(ceilMultiple(99, 100), 100);
}
