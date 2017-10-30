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
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/utility.hpp>

#include "Training.h"
#include "UCTNode.h"
#include "SGFParser.h"
#include "SGFTree.h"
#include "Random.h"

std::vector<TimeStep> Training::m_data{};

void Training::clear_training() {
    Training::m_data.clear();
}

void Training::record(GameState& state, const UCTNode& root) {
    auto step = TimeStep{};
    step.to_move = state.board.get_to_move();
    step.planes = Network::NNPlanes{};
    Network::gather_features(&state, step.planes);

    step.probabilities.resize((19 * 19) + 1);

    // Get total visit amount. We count rather
    // than trust the root to avoid ttable issues.
    auto sum_visits = 0.0;
    auto child = root.get_first_child();
    while (child != nullptr) {
        sum_visits += child->get_visits();
        child = child->get_sibling();
    }

    child = root.get_first_child();
    while (child != nullptr) {
        auto prob = child->get_visits() / sum_visits;
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            auto xy = state.board.get_xy(move);
            step.probabilities[xy.second * 19 + xy.first] = prob;
        } else {
            step.probabilities[19 * 19] = prob;
        }
        child = child->get_sibling();
    }

    m_data.emplace_back(step);
}

void Training::dump_training(int winner_color, const std::string& filename) {
    auto out = std::ofstream{filename, std::ofstream::out
                                       | std::ofstream::app};

    for (const auto& step : m_data) {
        // First output 16 times an input feature plane
        for (auto p = size_t{0}; p < 16; p++) {
            const auto& plane = step.planes[p];
            // Write it out as a string of hex characters
            for (auto bit = size_t{0}; bit + 3 < plane.size(); bit += 4) {
                auto hexbyte =  plane[bit]     << 3
                              | plane[bit + 1] << 2
                              | plane[bit + 2] << 1
                              | plane[bit + 3] << 0;
                out << std::hex << hexbyte;
            }
            // 361 % 4 = 1 so the last bit goes by itself
            assert(plane.size() % 4 == 1);
            out << plane[plane.size() - 1];
            out << std::dec << std::endl;
        }
        // The side to move planes can be compactly encoded into a single
        // bit, 0 = black to move.
        out << (step.to_move == FastBoard::BLACK ? "0" : "1") << std::endl;
        // Then a 362 long array of float probabilities
        for (auto it = begin(step.probabilities);
            it != end(step.probabilities); ++it) {
            out << *it;
            if (boost::next(it) != end(step.probabilities)) {
                out << " ";
            }
        }
        out << std::endl;
        // And the game result for the side to move
        if (step.to_move == winner_color) {
            out << "1";
        } else {
            out << "-1";
        }
        out << std::endl;
    }

    out.close();
}

void Training::process_game(GameState& state, size_t& train_pos, int who_won,
                            const std::vector<int>& tree_moves,
                            const std::string& out_filename) {
    clear_training();
    auto counter = size_t{0};
    state.rewind();

    do {
        auto to_move = state.get_to_move();
        auto move = tree_moves[counter];
        auto this_move = -1;

        // Detect if this SGF seems to be corrupted
        auto moves = state.generate_moves(to_move);
        auto moveseen = false;
        for(const auto& gen_move : moves) {
            if (gen_move == move) {
                if (move != FastBoard::PASS) {
                    // get x y coords for actual move
                    auto xy = state.board.get_xy(move);
                    this_move = (xy.second * 19) + xy.first;
                } else {
                    this_move = (19 * 19); // PASS
                }
                moveseen = true;
                break;
            }
        }

        if (!moveseen) {
            std::cout << "Mainline move not found: " << move << std::endl;
            return;
        }

        // Pick every 1/8th position.
        auto skip = Random::get_Rng()->randfix<8>();
        if (skip == 0) {
            auto step = TimeStep{};
            step.to_move = state.board.get_to_move();
            step.planes = Network::NNPlanes{};
            Network::gather_features(&state, step.planes);

            step.probabilities.resize((19 * 19) + 1);
            step.probabilities[this_move] = 1.0f;

            train_pos++;
            m_data.emplace_back(step);
        }

        counter++;
    } while (state.forward_move() && counter < tree_moves.size());

    dump_training(who_won, out_filename);
}

void Training::dump_supervised(const std::string& sgf_name,
                               const std::string& out_filename) {
    auto games = SGFParser::chop_all(sgf_name);
    auto gametotal = games.size();
    auto gamecount = size_t{0};
    auto train_pos = size_t{0};

    std::cout << "Total games in file: " << gametotal << std::endl;
    // Shuffle games around
    std::cout << "Shuffling...";
    std::shuffle(begin(games), end(games), *Random::get_Rng());
    std::cout << "done." << std::endl;

    while (gamecount < gametotal) {
        auto sgftree = std::make_unique<SGFTree>();
        try {
            sgftree->load_from_string(games[gamecount]);
        } catch (...) {
            continue;
        };

        auto tree_moves = sgftree->get_mainline();
        auto state =
            std::make_unique<GameState>(sgftree->follow_mainline_state());

        auto who_won = sgftree->get_winner();
        // Accept all komis and handicaps, but reject no usable result
        if (who_won == FastBoard::BLACK || who_won == FastBoard::WHITE) {
            // Our board size is hardcoded in several places
            if (state->board.get_boardsize() == 19) {
                process_game(*state, train_pos, who_won, tree_moves,
                             out_filename);
            }
        }

        gamecount++;
        if (gamecount % (1000) == 0) {
            std::cout << "Game " << gamecount
                      << ", " << train_pos << " positions" << std::endl;
        }
    }

    std::cout << "Dumped " << train_pos << " training positions." << std::endl;
}
