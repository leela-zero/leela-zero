/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Seth Troisi and contributors

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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include <boost/math/distributions/chi_squared.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "Random.h"
#include "Utils.h"

// Test should fail about this often from distribution not looking uniform.
// Increasing this allows better detection of bad RNG but increase the chance
// of test failure with acceptable RNG implemantation. On my system RNG seems
// to be a tiny bit not random and test fail about twice as often as predicted.
constexpr auto ALPHA = 0.0001;

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

bool rngBucketsLookRandom(double p, double alpha) {
    return p >= (alpha/2) && p <= (1-alpha/2);
}

TEST(UtilsTest, RandFix) {
    // Using seed = 0 results in pseudo-random seed.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    auto max = std::uint16_t{200};
    auto count = std::vector<short>(max, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[rng->randfix<200>()]++;
    }

    auto p = randomlyDistributedProbability(count, expected);
    EXPECT_PRED2(rngBucketsLookRandom, p, ALPHA);
}

TEST(UtilsTest, Randuint64_lastEightBits) {
    // Using seed = 0 results in pseudo-random seed.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    // Verify last 8 bits are random.
    auto max = std::uint16_t{128};
    auto count = std::vector<short>(max, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[rng->randuint64() & 127]++;
    }

    auto p = randomlyDistributedProbability(count, expected);
    EXPECT_PRED2(rngBucketsLookRandom, p, ALPHA);
}

TEST(UtilsTest, Randuint64_max) {
    // Using seed = 0 results in pseudo-random seed.
    auto rng = std::make_unique<Random>(0);

    auto expected = size_t{40};
    auto max = std::uint64_t{100};
    auto count = std::vector<short>(max, 0);
    for (auto i = size_t{0}; i < expected * max; i++) {
        count[rng->randuint64(max)]++;
    }

    auto p = randomlyDistributedProbability(count, expected);
    EXPECT_PRED2(rngBucketsLookRandom, p, ALPHA);
}
