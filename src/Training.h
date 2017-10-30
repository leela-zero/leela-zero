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

class OutputChunker {
public:
    OutputChunker(const std::string& basename, bool compress = false);
    ~OutputChunker();
    void append(const std::string& str);

    static constexpr size_t CHUNK_SIZE = 8192;
private:
    std::string gen_chunk_name() const;
    void flush_chunks();
    size_t m_step_count{0};
    size_t m_chunk_count{0};
    std::string m_buffer;
    std::string m_basename;
    bool m_compress{false};
};

class Training {
public:
    static void clear_training();
    static void dump_training(int winner_color,
                              const std::string& out_filename);
    static void record(GameState& state, const UCTNode& node);

    static void dump_supervised(const std::string& sgf_file,
                                const std::string& out_filename);
private:
    static void process_game(GameState& state, size_t& train_pos, int who_won,
                             const std::vector<int>& tree_moves,
                             OutputChunker& outchunker);
    static void dump_training(int winner_color,
                              OutputChunker& outchunker);
    static std::vector<TimeStep> m_data;
};

#endif
