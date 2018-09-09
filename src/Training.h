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

#ifndef TRAINING_H_INCLUDED
#define TRAINING_H_INCLUDED

#include "config.h"

#include <bitset>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "GameState.h"
#include "Network.h"
#include "UCTNode.h"

class TimeStep {
public:
    using BoardPlane = std::bitset<NUM_INTERSECTIONS>;
    using NNPlanes = std::vector<BoardPlane>;
    NNPlanes planes;
    std::vector<float> probabilities;
    int to_move;
    float stm_komi;
    float net_winrate;
    float root_uct_winrate;
    float child_uct_winrate;
    int bestmove_visits;
};

std::ostream& operator<< (std::ostream& stream, const TimeStep& timestep);
std::istream& operator>> (std::istream& stream, TimeStep& timestep);

class OutputChunker {
public:
    OutputChunker(const std::string& basename, bool compress = false);
    ~OutputChunker();
    void append(const std::string& str);

    // Group this many games in a batch.
    static constexpr size_t CHUNK_SIZE = 32;
private:
    std::string gen_chunk_name() const;
    void flush_chunks();
    size_t m_game_count{0};
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
    static void dump_debug(const std::string& out_filename);
    static void record(Network & network, GameState& state, UCTNode& node);

    static void dump_supervised(const std::string& sgf_file,
                                const std::string& out_filename);
    static void save_training(const std::string& filename);
    static void load_training(const std::string& filename);

private:
    static TimeStep::NNPlanes get_planes(const GameState* const state);
    static float get_stm_komi(const GameState* const state);
    static void process_game(GameState& state, size_t& train_pos, int who_won,
                             const std::vector<int>& tree_moves,
                             OutputChunker& outchunker);
    static void dump_training(int winner_color,
                              OutputChunker& outchunker);
    static void dump_debug(OutputChunker& outchunker);
    static void save_training(std::ofstream& out);
    static void load_training(std::ifstream& in);
    static std::vector<TimeStep> m_data;
};

#endif
