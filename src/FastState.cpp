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
#include "FastState.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "FastBoard.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

void FastState::init_game(int size, float komi) {
    board.reset_board(size);

    m_movenum = 0;

    m_komove = 0;
    std::fill(begin(m_lastmove), end(m_lastmove), 0);
    m_komi = komi;
    m_handicap = 0;
    m_passes = 0;

    return;
}

void FastState::set_komi(float komi) {
    m_komi = komi;
}

void FastState::reset_game(void) {
    reset_board();

    m_movenum = 0;
    m_passes = 0;
    m_handicap = 0;
    m_komove = 0;

    std::fill(begin(m_lastmove), end(m_lastmove), 0);
}

void FastState::reset_board(void) {
    board.reset_board(board.get_boardsize());
}

bool FastState::is_move_legal(int color, int vertex) {
    return vertex == FastBoard::PASS ||
           vertex == FastBoard::RESIGN ||
           (vertex != m_komove &&
                board.get_square(vertex) == FastBoard::EMPTY &&
                !board.is_suicide(vertex, color));
}

void FastState::play_pass(int color) {
    m_movenum++;

    std::rotate(rbegin(m_lastmove), rbegin(m_lastmove) + 1, rend(m_lastmove));
    m_lastmove[0] = FastBoard::PASS;

    if (board.m_tomove == color) {
        board.m_hash ^= 0xABCDABCDABCDABCDULL;
    }
    board.m_tomove = !color;

    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
    increment_passes();
    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
}

void FastState::play_move(int vertex) {
    play_move(board.m_tomove, vertex);
}

void FastState::play_move(int color, int vertex) {
    if (vertex != FastBoard::PASS) {
        int kosq = board.update_board(color, vertex);

        m_komove = kosq;
        std::rotate(rbegin(m_lastmove), rbegin(m_lastmove) + 1,
                    rend(m_lastmove));
        m_lastmove[0] = vertex;
        m_movenum++;

        if (board.m_tomove == color) {
            board.m_hash ^= 0xABCDABCDABCDABCDULL;
        }
        board.m_tomove = !color;

        if (get_passes() > 0) {
            board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
            set_passes(0);
            board.m_hash ^= Zobrist::zobrist_pass[0];
        }
    } else if (vertex == FastBoard::PASS) {
        play_pass(color);
    }
}

size_t FastState::get_movenum() const {
    return m_movenum;
}

int FastState::get_last_move(void) const {
    return m_lastmove.front();
}

int FastState::get_passes() const {
    return m_passes;
}

void FastState::set_passes(int val) {
    m_passes = val;
}

void FastState::increment_passes() {
    m_passes++;
    if (m_passes > 4) m_passes = 4;
}

int FastState::get_to_move() const {
    return board.m_tomove;
}

void FastState::set_to_move(int tom) {
    board.m_tomove = tom;
}

void FastState::display_state() {
    myprintf("\nPasses: %d            Black (X) Prisoners: %d\n",
             m_passes, board.get_prisoners(FastBoard::BLACK));
    if (board.black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }
    myprintf("    White (O) Prisoners: %d\n",
             board.get_prisoners(FastBoard::WHITE));

    board.display_board(get_last_move());
}

std::string FastState::move_to_text(int move) {
    return board.move_to_text(move);
}

float FastState::final_score() {
    FastState workstate(*this);
    return workstate.board.area_score(get_komi() + get_handicap());
}

float FastState::get_komi() const {
    return m_komi;
}

void FastState::set_handicap(int hcap) {
    m_handicap = hcap;
}

int FastState::get_handicap() const {
    return m_handicap;
}
