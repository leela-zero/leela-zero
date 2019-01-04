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
#include "KoState.h"

#include <cassert>
#include <algorithm>
#include <iterator>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"

void KoState::init_game(int size, float komi) {
    assert(size <= BOARD_SIZE);

    FastState::init_game(size, komi);

    m_ko_hash_history.clear();
    m_ko_hash_history.emplace_back(board.get_ko_hash());
}

bool KoState::superko() const {
    auto first = crbegin(m_ko_hash_history);
    auto last = crend(m_ko_hash_history);

    auto res = std::find(++first, last, board.get_ko_hash());

    return (res != last);
}

void KoState::reset_game() {
    FastState::reset_game();

    m_ko_hash_history.clear();
    m_ko_hash_history.push_back(board.get_ko_hash());
}

void KoState::play_move(int vertex) {
    play_move(board.get_to_move(), vertex);
}

void KoState::play_move(int color, int vertex) {
    if (vertex != FastBoard::RESIGN) {
        FastState::play_move(color, vertex);
    }
    m_ko_hash_history.push_back(board.get_ko_hash());
}
