/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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
#ifndef ZOBRIST_H_INCLUDED
#define ZOBRIST_H_INCLUDED

#include "config.h"

#include <array>
#include <cstdint>

#include "FastBoard.h"
#include "Random.h"

class Zobrist {
public:
    static constexpr auto zobrist_empty = 0x1234567887654321;
    static constexpr auto zobrist_blacktomove = 0xABCDABCDABCDABCD;

    static std::array<std::array<std::uint64_t, FastBoard::NUM_VERTICES>,     4> zobrist;
    static std::array<std::uint64_t, FastBoard::NUM_VERTICES>                    zobrist_ko;
    static std::array<std::array<std::uint64_t, FastBoard::NUM_VERTICES * 2>, 2> zobrist_pris;
    static std::array<std::uint64_t, 5>                                          zobrist_pass;

    static void init_zobrist(Random& rng);
};

#endif
