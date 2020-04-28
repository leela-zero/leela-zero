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

#include <array>
#include <cassert>

#include "FullBoard.h"
#include "Network.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

std::uint64_t FullBoard::calc_ko_hash() const {
    auto res = Zobrist::zobrist_empty;

    for (auto i = 0; i < m_numvertices; i++) {
        if (m_state[i] != INVAL) {
            res ^= Zobrist::zobrist[m_state[i]][i];
        }
    }

    /* Tromp-Taylor has positional superko */
    return res;
}

template<class Function>
std::uint64_t FullBoard::calc_hash(int komove, Function transform) const {
    auto res = Zobrist::zobrist_empty;

    for (auto i = 0; i < m_numvertices; i++) {
        if (m_state[i] != INVAL) {
            res ^= Zobrist::zobrist[m_state[i]][transform(i)];
        }
    }

    /* prisoner hashing is rule set dependent */
    res ^= Zobrist::zobrist_pris[0][m_prisoners[0]];
    res ^= Zobrist::zobrist_pris[1][m_prisoners[1]];

    if (m_tomove == BLACK) {
        res ^= Zobrist::zobrist_blacktomove;
    }

    res ^= Zobrist::zobrist_ko[transform(komove)];

    return res;
}

void FullBoard::record_position(int pos) {
    m_hash    ^= Zobrist::zobrist[m_state[pos]][pos];
    m_ko_hash ^= Zobrist::zobrist[m_state[pos]][pos];
}

void FullBoard::record_captures(const int color, const int captured_stones) {
    m_hash ^= Zobrist::zobrist_pris[color][m_prisoners[color]];
    m_prisoners[color] += captured_stones;
    m_hash ^= Zobrist::zobrist_pris[color][m_prisoners[color]];
}

std::uint64_t FullBoard::calc_hash(int komove) const {
    return calc_hash(komove, [](const auto vertex) { return vertex; });
}

std::uint64_t FullBoard::calc_symmetry_hash(int komove, int symmetry) const {
    return calc_hash(komove, [this, symmetry](const auto vertex) {
        if (vertex == NO_VERTEX) {
            return NO_VERTEX;
        } else {
            const auto newvtx = Network::get_symmetry(get_xy(vertex), symmetry, m_boardsize);
            return get_vertex(newvtx.first, newvtx.second);
        }
    });
}

std::uint64_t FullBoard::get_hash() const {
    return m_hash;
}

std::uint64_t FullBoard::get_ko_hash() const {
    return m_ko_hash;
}

void FullBoard::set_to_move(int tomove) {
    if (m_tomove != tomove) {
        m_hash ^= Zobrist::zobrist_blacktomove;
    }
    FastBoard::set_to_move(tomove);
}


void FullBoard::display_board(int lastmove) {
    FastBoard::display_board(lastmove);

    myprintf("Hash: %llX Ko-Hash: %llX\n\n", get_hash(), get_ko_hash());
}

void FullBoard::reset_board(int size) {
    FastBoard::reset_board(size);

    m_hash = calc_hash();
    m_ko_hash = calc_ko_hash();
}
