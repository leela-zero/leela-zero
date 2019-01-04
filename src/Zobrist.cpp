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

#include "config.h"
#include "Zobrist.h"
#include "Random.h"

std::array<std::array<std::uint64_t, FastBoard::NUM_VERTICES>,     4> Zobrist::zobrist;
std::array<std::uint64_t, FastBoard::NUM_VERTICES>                    Zobrist::zobrist_ko;
std::array<std::array<std::uint64_t, FastBoard::NUM_VERTICES * 2>, 2> Zobrist::zobrist_pris;
std::array<std::uint64_t, 5>                                          Zobrist::zobrist_pass;

void Zobrist::init_zobrist(Random& rng) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < FastBoard::NUM_VERTICES; j++) {
            Zobrist::zobrist[i][j] = rng.randuint64();
        }
    }

    for (int j = 0; j < FastBoard::NUM_VERTICES; j++) {
        Zobrist::zobrist_ko[j] = rng.randuint64();
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < FastBoard::NUM_VERTICES * 2; j++) {
            Zobrist::zobrist_pris[i][j] = rng.randuint64();
        }
    }

    for (int i = 0; i < 5; i++) {
        Zobrist::zobrist_pass[i]  = rng.randuint64();
    }
}
