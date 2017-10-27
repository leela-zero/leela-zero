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

#ifndef TRAINING_H_INCLUDED
#define TRAINING_H_INCLUDED

#include "config.h"
#include <string>
#include <utility>
#include "GameState.h"
#include "Network.h"

class TimeStep {
public:
    Network::NNPlanes planes;
    std::vector<float> probabilities;
    int to_move;
};

class Training {
public:
    static void clear_training();
    static void dump_training(int winner_color, std::string filename);
    static void record(GameState& state, UCTNode& node);
private:

    static std::vector<TimeStep> m_data;
};

#endif
