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
#include "Zobrist.h"
#include "Random.h"

std::array<std::array<std::uint64_t, FastBoard::MAXSQ>,     4> Zobrist::zobrist;
std::array<std::array<std::uint64_t, FastBoard::MAXSQ * 2>, 2> Zobrist::zobrist_pris;
std::array<std::uint64_t, 5>                                   Zobrist::zobrist_pass;

void Zobrist::init_zobrist(Random& rng) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < FastBoard::MAXSQ; j++) {
            Zobrist::zobrist[i][j]  = ((std::uint64_t)rng.randuint32()) << 32;
            Zobrist::zobrist[i][j] ^= (std::uint64_t)rng.randuint32();
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < FastBoard::MAXSQ * 2; j++) {
            Zobrist::zobrist_pris[i][j]  = ((std::uint64_t)rng.randuint32()) << 32;
            Zobrist::zobrist_pris[i][j] ^= (std::uint64_t)rng.randuint32();
        }
    }

    for (int i = 0; i < 5; i++) {
        Zobrist::zobrist_pass[i]  = ((std::uint64_t)rng.randuint32()) << 32;
        Zobrist::zobrist_pass[i] ^= (std::uint64_t)rng.randuint32();
    }
}
