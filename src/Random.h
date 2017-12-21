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

#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

#include "config.h"
#include <cstdint>
#include <limits>

/*
    Random number generator xoroshiro128+
*/
class Random {
public:
    Random() = delete;
    Random(std::uint64_t seed = 0);
    void seedrandom(std::uint64_t s);

    // random numbers from 0 to max
    template<int MAX>
    std::uint32_t randfix() {
        static_assert(0 < MAX &&
                     MAX < std::numeric_limits<std::uint32_t>::max(),
                     "randfix out of range");
        // Last bit isn't random, so don't use it in isolation. We specialize
        // this case.
        static_assert(MAX != 2, "don't isolate the LSB with xoroshiro128+");
        return gen() % MAX;
    }

    std::uint16_t randuint16(const std::uint16_t max);
    std::uint32_t randuint32(const std::uint32_t max);
    std::uint32_t randuint32();

    // random float from 0 to 1
    float randflt(void);

    // return the thread local RNG
    static Random& get_Rng(void);

    // UniformRandomBitGenerator interface
    using result_type = std::uint64_t;
    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }
    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }
    result_type operator()() {
        return gen();
    }

private:
    std::uint64_t gen(void);
    std::uint64_t m_s[2];
};

// Specialization for last bit: use sign test
template<>
inline std::uint32_t Random::randfix<2>() {
    return (gen() > (std::numeric_limits<std::uint64_t>::max() / 2));
}

#endif
