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

#include <array>
#include <cassert>

#include "FullBoard.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

int FullBoard::remove_string(int i) {
    int pos = i;
    int removed = 0;
    int color = m_square[i];

    do {
        m_hash    ^= Zobrist::zobrist[m_square[pos]][pos];
        m_ko_hash ^= Zobrist::zobrist[m_square[pos]][pos];

        m_square[pos] = EMPTY;
        m_parent[pos] = MAXSQ;

        remove_neighbour(pos, color);

        m_empty_idx[pos]      = m_empty_cnt;
        m_empty[m_empty_cnt]  = pos;
        m_empty_cnt++;

        m_hash    ^= Zobrist::zobrist[m_square[pos]][pos];
        m_ko_hash ^= Zobrist::zobrist[m_square[pos]][pos];

        removed++;
        pos = m_next[pos];
    } while (pos != i);

    return removed;
}

std::uint64_t FullBoard::calc_ko_hash(void) {
    auto res = std::uint64_t{0x1234567887654321ULL};

    for (int i = 0; i < m_maxsq; i++) {
        if (m_square[i] != INVAL) {
            res ^= Zobrist::zobrist[m_square[i]][i];
        }
    }

    m_ko_hash = res;

    /* Tromp-Taylor has positional superko */
    return res;
}

std::uint64_t FullBoard::calc_hash(void) {
    auto res = std::uint64_t{0x1234567887654321ULL};

    for (int i = 0; i < m_maxsq; i++) {
        if (m_square[i] != INVAL) {
            res ^= Zobrist::zobrist[m_square[i]][i];
        }
    }

    /* prisoner hashing is rule set dependent */
    res ^= Zobrist::zobrist_pris[0][m_prisoners[0]];
    res ^= Zobrist::zobrist_pris[1][m_prisoners[1]];

    if (m_tomove == BLACK) {
        res ^= 0xABCDABCDABCDABCDULL;
    }

    m_hash = res;

    return res;
}

std::uint64_t FullBoard::get_hash(void) const {
    return m_hash;
}

std::uint64_t FullBoard::get_ko_hash(void) const {
    return m_ko_hash;
}

int FullBoard::update_board(const int color, const int i) {
    assert(m_square[i] == EMPTY);

    m_hash ^= Zobrist::zobrist[m_square[i]][i];
    m_ko_hash ^= Zobrist::zobrist[m_square[i]][i];

    m_square[i] = (square_t)color;
    m_next[i] = i;
    m_parent[i] = i;
    m_libs[i] = count_pliberties(i);
    m_stones[i] = 1;

    m_hash ^= Zobrist::zobrist[m_square[i]][i];
    m_ko_hash ^= Zobrist::zobrist[m_square[i]][i];

    /* update neighbor liberties (they all lose 1) */
    add_neighbour(i, color);

    /* did we play into an opponent eye? */
    int eyeplay = (m_neighbours[i] & s_eyemask[!color]);

    int captured_sq;
    int captured_stones = 0;

    for (int k = 0; k < 4; k++) {
        int ai = i + m_dirs[k];

        if (m_square[ai] == !color) {
            if (m_libs[m_parent[ai]] <= 0) {
                int this_captured = remove_string(ai);
                captured_sq = ai;
                captured_stones += this_captured;
            }
        } else if (m_square[ai] == color) {
            int ip = m_parent[i];
            int aip = m_parent[ai];

            if (ip != aip) {
                if (m_stones[ip] >= m_stones[aip]) {
                    merge_strings(ip, aip);
                } else {
                    merge_strings(aip, ip);
                }
            }
        }
    }

    m_hash ^= Zobrist::zobrist_pris[color][m_prisoners[color]];
    m_prisoners[color] += captured_stones;
    m_hash ^= Zobrist::zobrist_pris[color][m_prisoners[color]];

    /* move last vertex in list to our position */
    int lastvertex = m_empty[--m_empty_cnt];
    m_empty_idx[lastvertex] = m_empty_idx[i];
    m_empty[m_empty_idx[i]] = lastvertex;

    /* check whether we still live (i.e. detect suicide) */
    if (m_libs[m_parent[i]] == 0) {
        assert(captured_stones == 0);
        remove_string(i);
    }

    /* check for possible simple ko */
    if (captured_stones == 1 && eyeplay) {
        return captured_sq;
    }

    return -1;
}

void FullBoard::display_board(int lastmove) {
    FastBoard::display_board(lastmove);

    myprintf("Hash: %llX Ko-Hash: %llX\n\n", get_hash(), get_ko_hash());
}

void FullBoard::reset_board(int size) {
    FastBoard::reset_board(size);

    calc_hash();
    calc_ko_hash();
}
