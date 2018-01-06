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

#include "GameState.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "UCTSearch.h"

void GameState::init_game(int size, float komi) {
    KoState::init_game(size, komi);

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.set_boardsize(board.get_boardsize());
    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

void GameState::reset_game() {
    KoState::reset_game();

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

bool GameState::forward_move(void) {
    if (game_history.size() > m_movenum + 1) {
        m_movenum++;
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];
        return true;
    } else {
        return false;
    }
}

bool GameState::undo_move(void) {
    if (m_movenum > 0) {
        m_movenum--;

        // this is not so nice, but it should work
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];

        // This also restores hashes as they're part of state
        return true;
    } else {
        return false;
    }
}

void GameState::rewind(void) {
    *(static_cast<KoState*>(this)) = *game_history[0];
    m_movenum = 0;
}

void GameState::play_move(int vertex) {
    play_move(get_to_move(), vertex);
}

void GameState::play_pass() {
    play_move(get_to_move(), FastBoard::PASS);
}

void GameState::play_move(int color, int vertex) {
    if (vertex != FastBoard::PASS && vertex != FastBoard::RESIGN) {
        KoState::play_move(color, vertex);
    } else if (vertex == FastBoard::PASS) {
        KoState::play_pass();
    } else if (vertex == FastBoard::RESIGN) {
        m_resigned = color;
    }

    // cut off any leftover moves from navigating
    game_history.resize(m_movenum);
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::play_textmove(std::string color, std::string vertex) {
    int who;
    int column, row;
    int boardsize = board.get_boardsize();

    if (color == "w" || color == "white") {
        who = FullBoard::WHITE;
    } else if (color == "b" || color == "black") {
        who = FullBoard::BLACK;
    } else return false;

    if (vertex.size() < 2) return 0;
    if (!std::isalpha(vertex[0])) return 0;
    if (!std::isdigit(vertex[1])) return 0;
    if (vertex[0] == 'i') return 0;

    if (vertex[0] >= 'A' && vertex[0] <= 'Z') {
        if (vertex[0] < 'I') {
            column = 25 + vertex[0] - 'A';
        } else {
            column = 25 + (vertex[0] - 'A')-1;
        }
    } else {
        if (vertex[0] < 'i') {
            column = vertex[0] - 'a';
        } else {
            column = (vertex[0] - 'a')-1;
        }
    }

    std::string rowstring(vertex);
    rowstring.erase(0, 1);
    std::istringstream parsestream(rowstring);

    parsestream >> row;
    row--;

    if (row >= boardsize) return false;
    if (column >= boardsize) return false;

    int move = board.get_vertex(column, row);

    set_to_move(who);
    play_move(move);

    return true;
}

void GameState::stop_clock(int color) {
    m_timecontrol.stop(color);
}

void GameState::start_clock(int color) {
    m_timecontrol.start(color);
}

void GameState::display_state() {
    FastState::display_state();

    m_timecontrol.display_times();
}

int GameState::who_resigned() const {
    return m_resigned;
}

bool GameState::has_resigned() const {
    return m_resigned != FastBoard::EMPTY;
}

TimeControl& GameState::get_timecontrol() {
    return m_timecontrol;
}

void GameState::set_timecontrol(int maintime, int byotime,
                                int byostones, int byoperiods) {
    TimeControl timecontrol(board.get_boardsize(), maintime, byotime,
                            byostones, byoperiods);

    m_timecontrol = timecontrol;
}

void GameState::set_timecontrol(TimeControl tmc) {
    m_timecontrol = tmc;
}

void GameState::adjust_time(int color, int time, int stones) {
    m_timecontrol.adjust_time(color, time, stones);
}

void GameState::anchor_game_history(void) {
    // handicap moves don't count in game history
    m_movenum = 0;
    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::set_fixed_handicap(int handicap) {
    if (!valid_handicap(handicap)) {
        return false;
    }

    int board_size = board.get_boardsize();
    int high = board_size >= 13 ? 3 : 2;
    int mid = board_size / 2;

    int low = board_size - 1 - high;
    if (handicap >= 2) {
        play_move(FastBoard::BLACK, board.get_vertex(low, low));
        play_move(FastBoard::BLACK, board.get_vertex(high, high));
    }

    if (handicap >= 3) {
        play_move(FastBoard::BLACK, board.get_vertex(high, low));
    }

    if (handicap >= 4) {
        play_move(FastBoard::BLACK, board.get_vertex(low, high));
    }

    if (handicap >= 5 && handicap % 2 == 1) {
        play_move(FastBoard::BLACK, board.get_vertex(mid, mid));
    }

    if (handicap >= 6) {
        play_move(FastBoard::BLACK, board.get_vertex(low, mid));
        play_move(FastBoard::BLACK, board.get_vertex(high, mid));
    }

    if (handicap >= 8) {
        play_move(FastBoard::BLACK, board.get_vertex(mid, low));
        play_move(FastBoard::BLACK, board.get_vertex(mid, high));
    }

    board.set_to_move(FastBoard::WHITE);

    anchor_game_history();

    set_handicap(handicap);

    return true;
}

int GameState::set_fixed_handicap_2(int handicap) {
    int board_size = board.get_boardsize();
    int low = board_size >= 13 ? 3 : 2;
    int mid = board_size / 2;
    int high = board_size - 1 - low;

    int interval = (high - mid) / 2;
    int placed = 0;

    while (interval >= 3) {
        for (int i = low; i <= high; i += interval) {
            for (int j = low; j <= high; j += interval) {
                if (placed >= handicap) return placed;
                if (board.get_square(i-1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_square(i-1, j) != FastBoard::EMPTY) continue;
                if (board.get_square(i-1, j+1) != FastBoard::EMPTY) continue;
                if (board.get_square(i, j-1) != FastBoard::EMPTY) continue;
                if (board.get_square(i, j) != FastBoard::EMPTY) continue;
                if (board.get_square(i, j+1) != FastBoard::EMPTY) continue;
                if (board.get_square(i+1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_square(i+1, j) != FastBoard::EMPTY) continue;
                if (board.get_square(i+1, j+1) != FastBoard::EMPTY) continue;
                play_move(FastBoard::BLACK, board.get_vertex(i, j));
                placed++;
            }
        }
        interval = interval / 2;
    }

    return placed;
}

bool GameState::valid_handicap(int handicap) {
    int board_size = board.get_boardsize();

    if (handicap < 2 || handicap > 9) {
        return false;
    }
    if (board_size % 2 == 0 && handicap > 4) {
        return false;
    }
    if (board_size == 7 && handicap > 4) {
        return false;
    }
    if (board_size < 7 && handicap > 0) {
        return false;
    }

    return true;
}

void GameState::place_free_handicap(int stones) {
    int limit = board.get_boardsize() * board.get_boardsize();
    if (stones > limit / 2) {
        stones = limit / 2;
    }

    int orgstones = stones;

    int fixplace = std::min(9, stones);

    set_fixed_handicap(fixplace);
    stones -= fixplace;

    stones -= set_fixed_handicap_2(stones);

    for (int i = 0; i < stones; i++) {
        auto search = std::make_unique<UCTSearch>(*this);

        int move = search->think(FastBoard::BLACK, UCTSearch::NOPASS);
        play_move(FastBoard::BLACK, move);
    }

    if (orgstones)  {
        board.set_to_move(FastBoard::WHITE);
    } else {
        board.set_to_move(FastBoard::BLACK);
    }

    anchor_game_history();

    set_handicap(orgstones);
}

const FullBoard& GameState::get_past_board(int moves_ago) const {
    assert(moves_ago >= 0 && (unsigned)moves_ago <= m_movenum);
    assert(m_movenum + 1 <= game_history.size());
    return game_history[m_movenum - moves_ago]->board;
}
