/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Gian-Carlo Pascutto
    Copyright (C) 2018 Seth Troisi

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

#include <boost/math/distributions/chi_squared.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>


#include "Random.h"
#include "Utils.h"

using namespace Utils;

TEST(UtilsTest, CeilMultiple) {
    // Equal to a multiple
    EXPECT_EQ(ceilMultiple(0, 1), (size_t)0);
    EXPECT_EQ(ceilMultiple(0, 3), (size_t)0);

    EXPECT_EQ(ceilMultiple(6,  1), (size_t)6);
    EXPECT_EQ(ceilMultiple(23, 1), (size_t)23);

    EXPECT_EQ(ceilMultiple(2, 2), (size_t)2);
    EXPECT_EQ(ceilMultiple(4, 2), (size_t)4);
    EXPECT_EQ(ceilMultiple(6, 2), (size_t)6);
    EXPECT_EQ(ceilMultiple(0, 3), (size_t)0);
    EXPECT_EQ(ceilMultiple(3, 3), (size_t)3);
    EXPECT_EQ(ceilMultiple(9, 3), (size_t)9);

    // Requires rounding up
    EXPECT_EQ(ceilMultiple(3, 5), (size_t)5);
    EXPECT_EQ(ceilMultiple(6, 5), (size_t)10);
    EXPECT_EQ(ceilMultiple(9, 5), (size_t)10);
    EXPECT_EQ(ceilMultiple(23, 5), (size_t)25);
    EXPECT_EQ(ceilMultiple(99, 100), (size_t)100);
}

bool NotRejectNull(double p, double alpha) {
    return p >= (alpha/2) && p <= (1-alpha/2);
}

double randomlyDistributedProbability(std::vector<short> values, double expected) {
    auto count = values.size();

    // h0: each number had a (1 / count) chance
    // Chi-square test that each bucket is a randomly distributed count

    // Variance of getting <v> at each iteration is Var[Bernoulli(1/count)]
    auto varIter = 1.0 / count - 1.0 / (count * count);
    // All rng are supposedly independant
    auto variance = count * expected * varIter;

    auto x = 0.0;
    for (const auto& observed : values) {
        auto error = observed - expected;
        auto t = (error * error) / variance;
        x += t;
    }

    auto degrees_of_freedom = count - 1;
    // test statistic of cdf(chi_squared_distribution(count - 1), q);
    return boost::math::gamma_p(degrees_of_freedom / 2.0, x / 2.0);
}

TEST(UtilsTest, Randuint16_max) {
    // 0 causes Random to use thread id.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    auto max = std::numeric_limits<std::uint16_t>::max();
    auto count = std::vector<short>(max + 1, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[rng->randuint16(max)]++;
    }

    auto p = randomlyDistributedProbability(count, expected);

    // Test should fail this often from distribution not looking uniform.
    auto alpha = 0.0001;
    EXPECT_PRED2(NotRejectNull, p, alpha);
}

TEST(UtilsTest, Randuint16_small) {
    // Using seed = 0 results in pseudo-random seed.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    auto max = std::uint16_t{100};
    auto count = std::vector<short>(max + 1, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[rng->randuint16(max)]++;
    }

    auto p = randomlyDistributedProbability(count, expected);

    // Test should fail this often from distribution not looking uniform.
    auto alpha = 0.0001;
    EXPECT_PRED2(NotRejectNull, p, alpha);
}

TEST(UtilsTest, Randuint64_partial) {
    // Using seed = 0 results in pseudo-random seed.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    // Verify last 8 bits are random.
    auto max = std::uint16_t{127};
    auto count = std::vector<short>(max + 1, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[rng->randuint64(max) & 127]++;
    }

    auto p = randomlyDistributedProbability(count, expected);

    // Test should fail this often from distribution not looking uniform.
    auto alpha = 0.0001;
    EXPECT_PRED2(NotRejectNull, p, alpha);
}

TEST(UtilsTest, Randflt) {
    // Using seed = 0 results in pseudo-random seed.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    auto max = std::uint16_t{200};
    auto count = std::vector<short>(max, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[int(max * rng->randflt())]++;
    }

    auto p = randomlyDistributedProbability(count, expected);

    // Test should fail this often from distribution not looking uniform.
    auto alpha = 0.0001;
    EXPECT_PRED2(NotRejectNull, p, alpha);
}
