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


#include "config.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include "zlib.h"

#include "Network.h"
#include "CPUPipe.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

// Symmetry helper
static std::array<std::array<int, BOARD_SQUARES>, Network::NUM_SYMMETRIES> symmetry_nn_idx_table;

void Network::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = 0; i < cpus; i++) {
        tg.add_task([this, &runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_output(state, Ensemble::RANDOM_SYMMETRY, -1, true);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

void Network::process_bn_var(std::vector<float>& weights, const float epsilon) {
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(2x2, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, WINOGRAD_TILE>{ 1.0,  0.0,  0.0,
                                                     0.5,  0.5,  0.5,
                                                     0.5, -0.5,  0.5,
                                                     0.0,  0.0,  1.0};
    auto temp = std::array<float, 12>{};

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto i = 0; i < 4; i++){
                for (auto j = 0; j < 3; j++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k < 3; k++) {
                        acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                    }
                    temp[i*3 + j] = acc;
                }
            }

            for (auto xi = 0; xi < 4; xi++) {
                for (auto nu = 0; nu < 4; nu++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k < 3; k++) {
                        acc += temp[xi*3 + k] * G[nu*3 + k];
                    }
                    U[xi * (4 * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o] = acc;
                }
            }
        }
    }

    return U;
}

std::pair<int, int> Network::load_v1_network(std::istream& wtfile) {
    // Count size of the network
    myprintf("Detecting residual layers...");
    // We are version 1 or 2
    if (m_value_head_not_stm) {
        myprintf("v%d...", 2);
    } else {
        myprintf("v%d...", 1);
    }
    // First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    auto line = std::string{};
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
            channels = count;
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        return {0, 0};
    }
    residual_blocks /= 8;
    myprintf("%d blocks.\n", residual_blocks);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    const auto plain_conv_layers = 1 + (residual_blocks * 2);
    const auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.cbegin();
        const auto ok = phrase_parse(it_line, line.cend(),
                                     *x3::float_, x3::space, weights);
        if (!ok || it_line != line.cend()) {
            myprintf("\nFailed to parse weight file. Error on line %d.\n",
                    linecount + 2); //+1 from version line, +1 from 0-indexing
            return {0,0};
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                m_conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                m_conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                m_batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                m_batchnorm_stddevs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            m_conv_pol_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 1) {
            m_conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            std::copy(cbegin(weights), cend(weights), begin(m_bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
            process_bn_var(weights);
            std::copy(cbegin(weights), cend(weights), begin(m_bn_pol_w2));
        } else if (linecount == plain_conv_wts + 4) {
            std::copy(cbegin(weights), cend(weights), begin(m_ip_pol_w));
        } else if (linecount == plain_conv_wts + 5) {
            std::copy(cbegin(weights), cend(weights), begin(m_ip_pol_b));
        } else if (linecount == plain_conv_wts + 6) {
            m_conv_val_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 7) {
            m_conv_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 8) {
            std::copy(cbegin(weights), cend(weights), begin(m_bn_val_w1));
        } else if (linecount == plain_conv_wts + 9) {
            process_bn_var(weights);
            std::copy(cbegin(weights), cend(weights), begin(m_bn_val_w2));
        } else if (linecount == plain_conv_wts + 10) {
            std::copy(cbegin(weights), cend(weights), begin(m_ip1_val_w));
        } else if (linecount == plain_conv_wts + 11) {
            std::copy(cbegin(weights), cend(weights), begin(m_ip1_val_b));
        } else if (linecount == plain_conv_wts + 12) {
            std::copy(cbegin(weights), cend(weights), begin(m_ip2_val_w));
        } else if (linecount == plain_conv_wts + 13) {
            std::copy(cbegin(weights), cend(weights), begin(m_ip2_val_b));
        }
        linecount++;
    }

    return {channels, static_cast<int>(residual_blocks)};
}

std::pair<int, int> Network::load_network_file(const std::string& filename) {
    // gzopen supports both gz and non-gz files, will decompress
    // or just read directly as needed.
    auto gzhandle = gzopen(filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        myprintf("Could not open weights file: %s\n", filename.c_str());
        return {0, 0};
    }
    // Stream the gz file in to a memory buffer stream.
    auto buffer = std::stringstream{};
    constexpr auto chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            myprintf("Failed to decompress or read: %s\n", filename.c_str());
            gzclose(gzhandle);
            return {0, 0};
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    gzclose(gzhandle);

    // Read format version
    auto line = std::string{};
    auto format_version = -1;
    if (std::getline(buffer, line)) {
        auto iss = std::stringstream{line};
        // First line is the file format version id
        iss >> format_version;
        if (iss.fail() || (format_version != 1 && format_version != 2)) {
            myprintf("Weights file is the wrong version.\n");
            return {0, 0};
        } else {
            // Version 2 networks are identical to v1, except
            // that they return the value for black instead of
            // the player to move. This is used by ELF Open Go.
            if (format_version == 2) {
                m_value_head_not_stm = true;
            } else {
                m_value_head_not_stm = false;
            }
            return load_v1_network(buffer);
        }
    }
    return {0, 0};
}

void Network::initialize(int playouts, const std::string & weightsfile) {
    m_nncache.set_size_from_playouts(playouts);
    // Prepare symmetry table
    for (auto s = 0; s < NUM_SYMMETRIES; ++s) {
        for (auto v = 0; v < BOARD_SQUARES; ++v) {
            const auto newvtx = get_symmetry({v % BOARD_SIZE, v / BOARD_SIZE}, s);
            symmetry_nn_idx_table[s][v] = (newvtx.second * BOARD_SIZE) + newvtx.first;
            assert(symmetry_nn_idx_table[s][v] >= 0 && symmetry_nn_idx_table[s][v] < BOARD_SQUARES);
        }
    }

    // Load network from file
    size_t channels, residual_blocks;
    std::tie(channels, residual_blocks) = load_network_file(weightsfile);
    if (channels == 0) {
        exit(EXIT_FAILURE);
    }

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    m_conv_weights[weight_index] =
        winograd_transform_f(m_conv_weights[weight_index],
                             channels, INPUT_CHANNELS);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < residual_blocks * 2; i++) {
        m_conv_weights[weight_index] =
            winograd_transform_f(m_conv_weights[weight_index],
                                 channels, channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    for (auto i = size_t{0}; i < m_conv_biases.size(); i++) {
        for (auto j = size_t{0}; j < m_batchnorm_means[i].size(); j++) {
            m_batchnorm_means[i][j] -= m_conv_biases[i][j];
            m_conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i < m_bn_val_w1.size(); i++) {
        m_bn_val_w1[i] -= m_conv_val_b[i];
        m_conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < m_bn_pol_w1.size(); i++) {
        m_bn_pol_w1[i] -= m_conv_pol_b[i];
        m_conv_pol_b[i] = 0.0f;
    }

    std::vector<ForwardPipe*> to_init;

#ifdef USE_OPENCL
    if (cfg_cpu_only) {
        myprintf("Initializing CPU-only evaluation.\n");
        m_forward.reset(new CPUPipe());
    } else {
        myprintf("Initializing OpenCL.\n");
        m_forward.reset(new OpenCLScheduler());
    }
#else
    myprintf("Initializing CPU-only evaluation.\n");
    m_forward.reset(new CPUPipe());
#endif

    to_init.emplace_back(m_forward.get());

#ifdef USE_OPENCL_SELFCHECK
    if (!cfg_cpu_only) {
        m_forward_cpu.reset(new CPUPipe());
        to_init.emplace_back(m_forward_cpu.get());
    }
#endif

    for (const auto& p : to_init) {
        p->initialize(channels);

        weight_index = 0;

        // Winograd filter transformation changes filter size to 4x4
        p->push_input_convolution(WINOGRAD_ALPHA, INPUT_CHANNELS,
            channels, m_conv_weights[weight_index],
            m_batchnorm_means[weight_index], m_batchnorm_stddevs[weight_index]);
        weight_index++;

        // residual blocks
        for (auto i = size_t{0}; i < residual_blocks; i++) {
            p->push_residual(WINOGRAD_ALPHA, channels, channels,
                             m_conv_weights[weight_index],
                             m_batchnorm_means[weight_index],
                             m_batchnorm_stddevs[weight_index],
                             m_conv_weights[weight_index + 1],
                             m_batchnorm_means[weight_index + 1],
                             m_batchnorm_stddevs[weight_index + 1]);
            weight_index += 2;
        }

        // Output head convolutions
        p->push_convolve(1, channels, OUTPUTS_POLICY, m_conv_pol_w);
        p->push_convolve(1, channels, OUTPUTS_VALUE, m_conv_val_w);
    }
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#endif
}

#ifdef USE_BLAS

template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU,
         size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases) {
    std::vector<float> output(outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }

    return output;
}

template <size_t spatial_size>
void batchnorm(const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddivs,
               const float* const eltwise = nullptr) {
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddiv * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

template<typename T>
T relative_difference(const T a, const T b) {
    // Handle NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<T>::max();
    }

    constexpr auto small_number = 1e-3f;
    auto fa = std::fabs(a);
    auto fb = std::fabs(b);

    if (fa > small_number && fb > small_number) {
        // Handle sign difference
        if ((a < 0) != (b < 0)) {
            return std::numeric_limits<T>::max();
        }
    } else {
        // Handle underflow
        fa = std::max(fa, small_number);
        fb = std::max(fb, small_number);
    }

    return fabs(fa - fb) / std::min(fa, fb);
}

void compare_net_outputs(std::vector<float>& data,
                         std::vector<float>& ref) {
    // We accept an error up to 5%, but output values
    // smaller than 1/1000th are "rounded up" for the comparison.
    constexpr auto relative_error = 5e-2f;
    for (auto idx = size_t{0}; idx < data.size(); ++idx) {
        const auto err = relative_difference(data[idx], ref[idx]);
        if (err > relative_error) {
            printf("Error in OpenCL calculation: expected %f got %f "
                   "(error=%f%%)\n", ref[idx], data[idx], err * 100.0);
            printf("Update your GPU drivers or reduce the amount of games "
                   "played simultaneously.\n");
            throw std::runtime_error("OpenCL self-check mismatch.");
        }
    }
}
#endif

std::vector<float> softmax(const std::vector<float>& input,
                           const float temperature = 1.0f) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}

bool Network::probe_cache(const GameState* const state,
                          Network::Netresult& result) {
    if (m_nncache.lookup(state->board.get_hash(), result)) {
        return true;
    }
    // If we are not generating a self-play game, try to find
    // symmetries if we are in the early opening.
    if (!cfg_noise && !cfg_random_cnt
        && state->get_movenum()
           < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
        for (auto sym = 0; sym < Network::NUM_SYMMETRIES; ++sym) {
            if (sym == Network::IDENTITY_SYMMETRY) {
                continue;
            }
            const auto hash = state->get_symmetry_hash(sym);
            if (m_nncache.lookup(hash, result)) {
                decltype(result.policy) corrected_policy;
                corrected_policy.reserve(BOARD_SQUARES);
                for (auto idx = size_t{0}; idx < BOARD_SQUARES; ++idx) {
                    const auto sym_idx = symmetry_nn_idx_table[sym][idx];
                    corrected_policy.emplace_back(result.policy[sym_idx]);
                }
                result.policy = std::move(corrected_policy);
                return true;
            }
        }
    }

    return false;
}

Network::Netresult Network::get_output(
    const GameState* const state, const Ensemble ensemble,
    const int symmetry, const bool skip_cache) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (!skip_cache) {
        // See if we already have this in the cache.
        if (probe_cache(state, result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        for (auto sym = 0; sym < NUM_SYMMETRIES; ++sym) {
            auto tmpresult = get_output_internal(state, sym);
            result.winrate += tmpresult.winrate / static_cast<float>(NUM_SYMMETRIES);
            result.policy_pass += tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);

            for (auto idx = size_t{0}; idx < BOARD_SQUARES; idx++) {
                result.policy[idx] += tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
    }

    // v2 format (ELF Open Go) returns black value, not stm
    if (m_value_head_not_stm) {
        if (state->board.get_to_move() == FastBoard::WHITE) {
            result.winrate = 1.0f - result.winrate;
        }
    }

    // Insert result into cache.
    m_nncache.insert(state->board.get_hash(), result);

    return result;
}

Network::Netresult Network::get_output_internal(
    const GameState* const state, const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    const auto input_data = gather_features(state, symmetry);
    std::vector<float> policy_data(OUTPUTS_POLICY * width * height);
    std::vector<float> value_data(OUTPUTS_VALUE * width * height);
    m_forward->forward(input_data, policy_data, value_data);
#ifdef USE_OPENCL_SELFCHECK
    // Both implementations are available, self-check the OpenCL driver by
    // running both with a probability of 1/2000.
    if (m_forward_cpu != nullptr && Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0) {
        auto cpu_policy_data = std::vector<float>(policy_data.size());
        auto cpu_value_data = std::vector<float>(value_data.size());
        m_forward_cpu->forward(input_data, cpu_policy_data, cpu_value_data);
        compare_net_outputs(policy_data, cpu_policy_data);
        compare_net_outputs(value_data, cpu_value_data);
    }
#endif

    // Get the moves
    batchnorm<BOARD_SQUARES>(OUTPUTS_POLICY, policy_data,
        m_bn_pol_w1.data(), m_bn_pol_w2.data());
    const auto policy_out =
        innerproduct<OUTPUTS_POLICY * BOARD_SQUARES, BOARD_SQUARES + 1, false>(
            policy_data, m_ip_pol_w, m_ip_pol_b);
    const auto outputs = softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    batchnorm<BOARD_SQUARES>(OUTPUTS_VALUE, value_data,
        m_bn_val_w1.data(), m_bn_val_w2.data());
    const auto winrate_data =
        innerproduct<BOARD_SQUARES, 256, true>(value_data, m_ip1_val_w, m_ip1_val_b);
    const auto winrate_out =
        innerproduct<256, 1, false>(winrate_data, m_ip2_val_w, m_ip2_val_b);

    // Map TanH output range [-1..1] to [0..1] range
    const auto winrate = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    Netresult result;

    for (auto idx = size_t{0}; idx < BOARD_SQUARES; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = outputs[idx];
    }

    result.policy_pass = outputs[BOARD_SQUARES];
    result.winrate = winrate;

    return result;
}

void Network::show_heatmap(const FastState* const state,
                           const Netresult& result,
                           const bool topmoves) {
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            auto policy = 0;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_square(vertex) == FastBoard::EMPTY) {
                policy = result.policy[y * BOARD_SIZE + x] * 1000;
            }

            line += boost::str(boost::format("%3d ") % policy);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    const auto pass_policy = int(result.policy_pass * 1000);
    myprintf("pass: %d\n", pass_policy);
    myprintf("winrate: %f\n", result.winrate);

    if (topmoves) {
        std::vector<Network::PolicyVertexPair> moves;
        for (auto i=0; i < BOARD_SQUARES; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_square(vertex) == FastBoard::EMPTY) {
                moves.emplace_back(result.policy[i], vertex);
            }
        }
        moves.emplace_back(result.policy_pass, FastBoard::PASS);

        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        size_t tried = 0;
        while (cum < 0.85f && tried < moves.size()) {
            if (moves[tried].first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    moves[tried].first,
                    state->board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < BOARD_SQUARES; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_square(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = float(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = float(true);
        }
    }
}

std::vector<float> Network::gather_features(const GameState* const state,
                                            const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    auto input_data = std::vector<float>(INPUT_CHANNELS * BOARD_SQUARES);

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    const auto black_it = blacks_move ?
                          begin(input_data) :
                          begin(input_data) + INPUT_MOVES * BOARD_SQUARES;
    const auto white_it = blacks_move ?
                          begin(input_data) + INPUT_MOVES * BOARD_SQUARES :
                          begin(input_data);
    const auto to_move_it = blacks_move ?
        begin(input_data) + 2 * INPUT_MOVES * BOARD_SQUARES :
        begin(input_data) + (2 * INPUT_MOVES + 1) * BOARD_SQUARES;

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_board(h),
                              black_it + h * BOARD_SQUARES,
                              white_it + h * BOARD_SQUARES,
                              symmetry);
    }

    std::fill(to_move_it, to_move_it + BOARD_SQUARES, float(true));

    return input_data;
}

std::pair<int, int> Network::get_symmetry(const std::pair<int, int>& vertex,
                                          const int symmetry,
                                          const int board_size) {
    auto x = vertex.first;
    auto y = vertex.second;
    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    if ((symmetry & 4) != 0) {
        std::swap(x, y);
    }

    if ((symmetry & 2) != 0) {
        x = board_size - x - 1;
    }

    if ((symmetry & 1) != 0) {
        y = board_size - y - 1;
    }

    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry != IDENTITY_SYMMETRY || vertex == std::make_pair(x, y));
    return {x, y};
}
