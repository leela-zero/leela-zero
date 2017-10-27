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
#include <iostream>
#include <fstream>
#include <boost/utility.hpp>

#include "Training.h"
#include "UCTNode.h"

std::vector<TimeStep> Training::m_data{};

void Training::clear_training() {
    Training::m_data.clear();
}

void Training::record(GameState& state, UCTNode& root) {
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

void Training::dump_training(int winner_color, std::string filename) {
    auto out = std::ofstream{filename, std::ofstream::out
                                       | std::ofstream::app};

    for (const auto& step : m_data) {
        // First output 18 bit vector planes
        for (const auto& plane : step.planes) {
            for (auto bit = size_t{0}; bit < plane.size(); ++bit) {
                if (plane[bit]) {
                    out << "1";
                } else {
                    out << "0";
                }
                if (bit != plane.size() - 1) {
                    out << " ";
                }
            }
            out << std::endl;
        }
        // Then a 362 long array of probabilities
        for (auto it = begin(step.probabilities);
            it != end(step.probabilities); ++it) {
            out << *it;
            if (boost::next(it) != end(step.probabilities)) {
                out << " ";
            }
        }
        out << std::endl;
        // And the game result
        if (step.to_move == winner_color) {
            out << "1";
        } else {
            out << "-1";
        }
        out << std::endl;
    }

    out.close();
}
