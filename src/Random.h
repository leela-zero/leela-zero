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
#include <limits>

/*
    Random number generator xoroshiro128+
*/
class Random {
public:
    Random(int seed = -1);
    void seedrandom(uint32 s);

    // random numbers from 0 to max
    template<int MAX>
    uint32 randfix() {
        static_assert(0 < MAX && MAX < std::numeric_limits<uint32>::max(),
                     "randfix out of range");
        // Last bit isn't random, so don't use it in isolation. We specialize
        // this case.
        static_assert(MAX != 2, "don't isolate the LSB with xoroshiro128+");
        return random() % MAX;
    }

    uint16 randuint16(const uint16 max);
    uint32 randuint32(const uint32 max);
    uint32 randuint32();

    // random float from 0 to 1
    float randflt(void);

    // return the thread local RNG
    static Random* get_Rng(void);

    // UniformRandomBitGenerator interface
    using result_type = uint64;
    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }
    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }
    result_type operator()() {
        return random();
    }

private:
    uint64 random(void);
    uint64 m_s[2];
};

// Specialization for last bit: use sign test
template<>
inline uint32 Random::randfix<2>() {
    return (random() > (std::numeric_limits<uint64>::max() / 2));
}

#endif
