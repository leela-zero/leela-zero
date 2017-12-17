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
#include <cctype>
#include <string>
#include <sstream>
#include <algorithm>
#include <array>
#include <bitset>
#include <utility>
#include <deque>

#include "config.h"

#include "KoState.h"
#include "GameState.h"
#include "FullBoard.h"
#include "UCTSearch.h"
#include "Zobrist.h"
#include "Random.h"
#include "Utils.h"

void GameState::init_game(int size, float komi) {
    KoState::init_game(size, komi);

    game_history.clear();
    append_to_gamehistory();

    m_boardplanes.clear();
    update_boardplanes();

    m_timecontrol.set_boardsize(board.get_boardsize());
    m_timecontrol.reset_clocks();

    return;
};

void GameState::reset_game() {
    KoState::reset_game();

    game_history.clear();
    append_to_gamehistory();

    m_boardplanes.clear();
    update_boardplanes();

    m_timecontrol.reset_clocks();
}

bool GameState::forward_move(void) {
    if (m_history_enabled && game_history.size() > m_movenum + 1) {
        m_movenum++;
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];
        return true;
    } else {
        return false;
    }
}

bool GameState::undo_move(void) {
    if (m_history_enabled && m_movenum > 0) {
        m_movenum--;

        // don't actually delete it!
        //game_history.pop_back();

        // this is not so nice, but it should work
        *(static_cast<KoState*>(this)) = *game_history[m_movenum];

        // This also restores hashes as they're part of state
        return true;
    } else {
        return false;
    }
}

void GameState::rewind(void) {
    if (m_history_enabled) {
        *(static_cast<KoState*>(this)) = *game_history[0];
        m_movenum = 0;
    }
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
    } else {
        KoState::play_pass();
        if (vertex == FastBoard::RESIGN) {
            std::rotate(rbegin(m_lastmove), rbegin(m_lastmove) + 1,
                        rend(m_lastmove));
            m_lastmove[0] = vertex;
            m_last_was_capture = false;
        }
    }

    // cut off any leftover moves from navigating
    if (m_history_enabled) {
        game_history.resize(m_movenum);
        append_to_gamehistory();
    }

    m_boardplanes.resize(m_movenum);
    update_boardplanes();
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

    play_move(who, move);

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
    append_to_gamehistory();

    m_boardplanes.clear();
    update_boardplanes();
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

void GameState::disable_history() {
    m_history_enabled = false;
    game_history.clear();
}

void GameState::append_to_gamehistory() {
    if (m_history_enabled) {
        game_history.emplace_back(std::make_shared<KoState>(*this));
    }
}

void GameState::update_boardplanes() {
    if (m_boardplanes.size() == Network::INPUT_MOVES) {
        m_boardplanes.pop_back();
    }

    Network::BoardPlane black, white;
    state_to_board_plane(black, white);
    m_boardplanes.emplace_front(std::make_pair(black, white));
}

std::pair<Network::BoardPlane*, Network::BoardPlane*> GameState::get_boardplanes(int moves_ago) {
    assert(moves_ago < NETWORK::INPUT_MOVES);
    assert(moves_ago < m_boardplanes.size());
    return make_pair(&m_boardplanes[moves_ago].first, &m_boardplanes[moves_ago].second);
}

void GameState::state_to_board_plane(Network::BoardPlane& black, Network::BoardPlane& white) {
    auto idx = 0;
    for (int j = 0; j < 19; j++) {
        for(int i = 0; i < 19; i++) {
            int vtx = board.get_vertex(i, j);
            FastBoard::square_t color = board.get_square(vtx);
            if (color != FastBoard::EMPTY) {
                if (color == FastBoard::BLACK) {
                    black[idx] = true;
                } else {
                    white[idx] = true;
                }
            }
            idx++;
        }
    }
}
