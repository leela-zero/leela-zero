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

#include <assert.h>
#include <stdlib.h>
#include <ctype.h>
#include <string>
#include <algorithm>

#include "config.h"

#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"

void KoState::init_game(int size, float komi) {
    assert(size <= FastBoard::MAXBOARDSIZE);

    FastState::init_game(size, komi);

    ko_hash_history.clear();
    hash_history.clear();

    ko_hash_history.push_back(board.calc_ko_hash());
    hash_history.push_back(board.calc_hash());
}

bool KoState::superko(void) {
    auto first = crbegin(ko_hash_history);
    auto last = crend(ko_hash_history);

    auto res = std::find(++first, last, board.ko_hash);

    return (res != last);
}

bool KoState::superko(uint64 newhash) {
    auto first = crbegin(ko_hash_history);
    auto last = crend(ko_hash_history);

    auto res = std::find(first, last, newhash);

    return (res != last);
}

void KoState::reset_game() {
    FastState::reset_game();

    ko_hash_history.clear();
    hash_history.clear();

    ko_hash_history.push_back(board.calc_ko_hash());
    hash_history.push_back(board.calc_hash());
}

void KoState::play_pass(void) {
    FastState::play_pass();

    ko_hash_history.push_back(board.ko_hash);
    hash_history.push_back(board.hash);
}

void KoState::play_move(int vertex) {
    play_move(board.get_to_move(), vertex);
}

void KoState::play_move(int color, int vertex) {
    if (vertex != FastBoard::PASS && vertex != FastBoard::RESIGN) {
        FastState::play_move(color, vertex);

        ko_hash_history.push_back(board.ko_hash);
        hash_history.push_back(board.hash);
    } else {
        play_pass();
    }
}
