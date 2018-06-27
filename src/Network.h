/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "NNCache.h"
#include "FastState.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif

class GameState;

class Network {
public:
    static constexpr auto NUM_SYMMETRIES = 8;
    static constexpr auto IDENTITY_SYMMETRY = 0;
    enum Ensemble {
        DIRECT, RANDOM_SYMMETRY, AVERAGE
    };
    using ScoreVertexPair = std::pair<float,int>;
    using Netresult = NNCache::Netresult;

    Netresult get_scored_moves(const GameState* const state,
                                      const Ensemble ensemble,
                                      const int symmetry = -1,
                                      const bool skip_cache = false);

    static constexpr auto INPUT_MOVES = 8;
    static constexpr auto INPUT_CHANNELS = 2 * INPUT_MOVES + 2;
    static constexpr auto OUTPUTS_POLICY = 2;
    static constexpr auto OUTPUTS_VALUE = 1;

    // Winograd filter transformation changes 3x3 filters to 4x4
    static constexpr auto WINOGRAD_ALPHA = 4;
    static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

    void initialize(int playouts, const std::string & weightsfile);
    void benchmark(const GameState * const state,
                          const int iterations = 1600);
    static void show_heatmap(const FastState * const state,
                             const Netresult & netres, const bool topmoves);

    static std::vector<net_t> gather_features(const GameState* const state,
                                              const int symmetry);
    static std::pair<int, int> get_symmetry(const std::pair<int, int>& vertex,
                                            const int symmetry,
                                            const int board_size = BOARD_SIZE);
private:
    std::pair<int, int> load_v1_network(std::istream& wtfile);
    std::pair<int, int> load_network_file(const std::string& filename);
    void process_bn_var(std::vector<float>& weights,
                               const float epsilon = 1e-5f);

    std::vector<float> winograd_transform_f(const std::vector<float>& f,
        const int outputs, const int channels);
    std::vector<float> zeropad_U(const std::vector<float>& U,
        const int outputs, const int channels,
        const int outputs_pad, const int channels_pad);
    void winograd_transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C);
    void winograd_transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K);
    void winograd_convolve3(const int outputs,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output);
    void winograd_sgemm(const std::vector<float>& U,
                               const std::vector<float>& V,
                               std::vector<float>& M, const int C, const int K);
    Netresult get_scored_moves_internal(const GameState* const state,
                                               const int symmetry);

    static void fill_input_plane_pair(const FullBoard& board,
                                      std::vector<net_t>::iterator black,
                                      std::vector<net_t>::iterator white,
                                      const int symmetry);

    bool probe_cache(const GameState* const state, Network::Netresult& result);
#if defined(USE_BLAS)
    void forward_cpu(const std::vector<float>& input,
                            std::vector<float>& output_pol,
                            std::vector<float>& output_val);

#endif

#ifdef USE_OPENCL
    OpenCLScheduler m_opencl;
#endif
    NNCache m_nncache;

    // Input + residual block tower
    std::vector<std::vector<float>> conv_weights;
    std::vector<std::vector<float>> conv_biases;
    std::vector<std::vector<float>> batchnorm_means;
    std::vector<std::vector<float>> batchnorm_stddivs;

    // Policy head
    std::vector<float> conv_pol_w;
    std::vector<float> conv_pol_b;
    std::array<float, 2> bn_pol_w1;
    std::array<float, 2> bn_pol_w2;

    std::array<float, (BOARD_SQUARES + 1) * BOARD_SQUARES * 2> ip_pol_w;
    std::array<float, BOARD_SQUARES + 1> ip_pol_b;

    // Value head
    std::vector<float> conv_val_w;
    std::vector<float> conv_val_b;
    std::array<float, 1> bn_val_w1;
    std::array<float, 1> bn_val_w2;

    std::array<float, BOARD_SQUARES * 256> ip1_val_w;
    std::array<float, 256> ip1_val_b;

    std::array<float, 256> ip2_val_w;
    std::array<float, 1> ip2_val_b;
    bool value_head_not_stm;
};
#endif
