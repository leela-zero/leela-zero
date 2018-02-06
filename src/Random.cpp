/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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

#include "config.h"
#include "Random.h"

#include <climits>
#include <cstdint>
#include <thread>
#include <random>

#include "GTP.h"
#include "Utils.h"

Random& Random::get_Rng(void) {
    static thread_local Random s_rng{0};
    return s_rng;
}

Random::Random(std::uint64_t seed) {
    if (seed == 0) {
        size_t thread_id =
            std::hash<std::thread::id>()(std::this_thread::get_id());
        seedrandom(cfg_rng_seed ^ (std::uint64_t)thread_id);
    } else {
        seedrandom(seed);
    }
}

// This is xoroshiro128+.
// Note that the last bit isn't entirely random, so don't use it,
// if possible.
std::uint64_t Random::gen(void) {
    const std::uint64_t s0 = m_s[0];
    std::uint64_t s1 = m_s[1];
    const std::uint64_t result = s0 + s1;

    s1 ^= s0;
    m_s[0] = Utils::rotl(s0, 55) ^ s1 ^ (s1 << 14);
    m_s[1] = Utils::rotl(s1, 36);

    return result;
}

std::uint64_t Random::randuint64(const uint64_t max) {
    const uint64_t inclusive_max = max - 1;
    return std::uniform_int_distribution<uint64_t>{0, inclusive_max}(*this);
}

std::uint64_t Random::randuint64() {
    return gen();
}

static std::uint64_t splitmix64(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

void Random::seedrandom(std::uint64_t seed) {
    // Initialize state of xoroshiro128+ by transforming the seed
    // with the splitmix64 algorithm.
    // As suggested by http://xoroshiro.di.unimi.it/xoroshiro128plus.c
    m_s[0] = splitmix64(seed);
    m_s[1] = splitmix64(m_s[0]);
}

