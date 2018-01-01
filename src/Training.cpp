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

#include "Training.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "FastBoard.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "Random.h"
#include "SGFParser.h"
#include "SGFTree.h"
#include "Timing.h"
#include "UCTNode.h"
#include "Utils.h"
#include "string.h"
#include "zlib.h"

std::vector<TimeStep> Training::m_data{};

std::string OutputChunker::gen_chunk_name(void) const {
    auto base = std::string{m_basename};
    base.append("." + std::to_string(m_chunk_count) + ".gz");
    return base;
}

OutputChunker::OutputChunker(const std::string& basename,
                             bool compress)
    : m_basename(basename), m_compress(compress) {
}

OutputChunker::~OutputChunker() {
    flush_chunks();
}

void OutputChunker::append(const std::string& str) {
    m_buffer.append(str);
    m_step_count++;
    if (m_step_count >= CHUNK_SIZE) {
        flush_chunks();
    }
}

void OutputChunker::flush_chunks() {
    if (m_compress) {
        auto chunk_name = gen_chunk_name();
        auto out = gzopen(chunk_name.c_str(), "wb9");

        auto in_buff_size = m_buffer.size();
        auto in_buff = std::make_unique<char[]>(in_buff_size);
        memcpy(in_buff.get(), m_buffer.data(), in_buff_size);

        auto comp_size = gzwrite(out, in_buff.get(), in_buff_size);
        if (!comp_size) {
            throw std::runtime_error("Error in gzip output");
        }
        Utils::myprintf("Writing chunk %d\n",  m_chunk_count);
        gzclose(out);
    } else {
        auto chunk_name = m_basename;
        auto flags = std::ofstream::out | std::ofstream::app;
        auto out = std::ofstream{chunk_name, flags};
        out << m_buffer;
        out.close();
    }

    m_buffer.clear();
    m_chunk_count++;
    m_step_count = 0;
}

void Training::clear_training() {
    Training::m_data.clear();
}

void Training::record(GameState& state, UCTNode& root) {
    auto step = TimeStep{};
    step.to_move = state.board.get_to_move();
    step.planes = Network::NNPlanes{};
    Network::gather_features(&state, step.planes);

    auto result =
        Network::get_scored_moves(&state, Network::Ensemble::DIRECT, 0);
    step.net_winrate = result.second;

    const auto& best_node = root.get_best_root_child(step.to_move);
    step.root_uct_winrate = root.get_eval(step.to_move);
    step.child_uct_winrate = best_node.get_eval(step.to_move);
    step.bestmove_visits = best_node.get_visits();

    step.probabilities.resize((19 * 19) + 1);

    // Get total visit amount. We count rather
    // than trust the root to avoid ttable issues.
    auto sum_visits = 0.0;
    for (const auto& child : root.get_children()) {
        sum_visits += child->get_visits();
    }

    // In a terminal position (with 2 passes), we can have children, but we
    // will not able to accumulate search results on them because every attempt
    // to evaluate will bail immediately. So in this case there will be 0 total
    // visits, and we should not construct the (non-existent) probabilities.
    if (sum_visits <= 0.0) {
        return;
    }

    for (const auto& child : root.get_children()) {
        auto prob = static_cast<float>(child->get_visits() / sum_visits);
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            auto xy = state.board.get_xy(move);
            step.probabilities[xy.second * 19 + xy.first] = prob;
        } else {
            step.probabilities[19 * 19] = prob;
        }
    }

    m_data.emplace_back(step);
}

void Training::dump_training(int winner_color, const std::string& filename) {
    auto chunker = OutputChunker{filename, true};
    dump_training(winner_color, chunker);
}

void Training::dump_training(int winner_color, OutputChunker& outchunk) {
    for (const auto& step : m_data) {
        auto out = std::stringstream{};
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
            if (next(it) != end(step.probabilities)) {
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
        outchunk.append(out.str());
    }
}

void Training::dump_debug(const std::string& filename) {
    auto chunker = OutputChunker{filename, true};
    dump_debug(chunker);
}

void Training::dump_debug(OutputChunker& outchunk) {
    {
        auto out = std::stringstream{};
        out << "2" << std::endl; // File format version
        out << cfg_resignpct << " " << cfg_weightsfile << std::endl;
        outchunk.append(out.str());
    }
    for (const auto& step : m_data) {
        auto out = std::stringstream{};
        out << step.net_winrate
            << " " << step.root_uct_winrate
            << " " << step.child_uct_winrate
            << " " << step.bestmove_visits << std::endl;
        outchunk.append(out.str());
    }
}

void Training::process_game(GameState& state, size_t& train_pos, int who_won,
                            const std::vector<int>& tree_moves,
                            OutputChunker& outchunker) {
    clear_training();
    auto counter = size_t{0};
    state.rewind();

    do {
        auto to_move = state.get_to_move();
        auto move_vertex = tree_moves[counter];
        auto move_idx = size_t{0};

        // Detect if this SGF seems to be corrupted
        if (!state.is_move_legal(to_move, move_vertex)) {
            std::cout << "Mainline move not found: " << move_vertex << std::endl;
            return;
        }

        if (move_vertex != FastBoard::PASS) {
            // get x y coords for actual move
            auto xy = state.board.get_xy(move_vertex);
            move_idx = (xy.second * 19) + xy.first;
        } else {
            move_idx = 19 * 19; // PASS
        }


        // Pick every 1/SKIP_SIZE th position.
        auto skip = Random::get_Rng().randfix<SKIP_SIZE>();
        if (skip == 0) {
            auto step = TimeStep{};
            step.to_move = to_move;
            step.planes = Network::NNPlanes{};
            Network::gather_features(&state, step.planes);

            step.probabilities.resize((19 * 19) + 1);
            step.probabilities[move_idx] = 1.0f;

            train_pos++;
            m_data.emplace_back(step);
        }

        counter++;
    } while (state.forward_move() && counter < tree_moves.size());

    dump_training(who_won, outchunker);
}

void Training::dump_supervised(const std::string& sgf_name,
                               const std::string& out_filename) {
    auto outchunker = OutputChunker{out_filename, true};
    auto games = SGFParser::chop_all(sgf_name);
    auto gametotal = games.size();
    auto train_pos = size_t{0};

    std::cout << "Total games in file: " << gametotal << std::endl;
    // Shuffle games around
    std::cout << "Shuffling...";
    std::shuffle(begin(games), end(games), Random::get_Rng());
    std::cout << "done." << std::endl;

    Time start;
    // Loop over the database multiple times. We will select different
    // positions from each game on every pass.
    for (auto repeat = size_t{0}; repeat < SKIP_SIZE; repeat++) {
        for (auto gamecount = size_t{0}; gamecount < gametotal; gamecount++) {
            auto sgftree = std::make_unique<SGFTree>();
            try {
                sgftree->load_from_string(games[gamecount]);
            } catch (...) {
                continue;
            };

            if (gamecount > 0 && gamecount % 1000 == 0) {
                Time elapsed;
                auto elapsed_s = Time::timediff_seconds(start, elapsed);
                Utils::myprintf("Game %5d, %5d positions in %5.2f seconds -> %d pos/s\n",
                    gamecount, train_pos, elapsed_s, (int)(train_pos / elapsed_s));
            }

            auto tree_moves = sgftree->get_mainline();
            // Empty game or couldn't be parsed?
            if (tree_moves.size() == 0) {
                continue;
            }

            auto who_won = sgftree->get_winner();
            // Accept all komis and handicaps, but reject no usable result
            if (who_won != FastBoard::BLACK && who_won != FastBoard::WHITE) {
                continue;
            }

            auto state =
                std::make_unique<GameState>(sgftree->follow_mainline_state());
            // Our board size is hardcoded in several places
            if (state->board.get_boardsize() != 19) {
                continue;
            }

            process_game(*state, train_pos, who_won, tree_moves,
                        outchunker);
        }
    }

    std::cout << "Dumped " << train_pos << " training positions." << std::endl;
}
