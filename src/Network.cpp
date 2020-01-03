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

#include "config.h"

#include <algorithm>
#include <array>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/utility.hpp>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include "CPUPipe.h"
#include "Network.h"
#include "zlib.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

// Symmetry helper
static std::array<std::array<int, NUM_INTERSECTIONS>, Network::NUM_SYMMETRIES>
    symmetry_nn_idx_table;

float Network::benchmark_time(const int centiseconds) {
    const auto cpus = cfg_num_threads;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    GameState state;
    state.init_game(BOARD_SIZE, TRAINED_UNIT_KOMI);

    // As a sanity run, try one run with self check.
    // Isn't enough to guarantee correctness but better than nothing,
    // plus for large nets self-check takes a while (1~3 eval per second)
    get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false, true, true);

    const Time start;
    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, start, centiseconds, state]() {
            while (true) {
                runcount++;
                get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false);
                const Time end;
                const auto elapsed = Time::timediff_centis(start, end);
                if (elapsed >= centiseconds) {
                    break;
                }
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_centis(start, end);
    return 100.0f * runcount.load() / elapsed;
}

void Network::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_output(state, Ensemble::RANDOM_SYMMETRY, -1, false);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

template <class container>
void process_bn_var(container& weights) {
    constexpr auto epsilon = 1e-5f;
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, 3 * WINOGRAD_ALPHA>{
         1.0f,         0.0f,        0.0f,
        -2.0f / 3.0f, -SQ2 / 3.0f, -1.0f / 3.0f,
        -2.0f / 3.0f,  SQ2 / 3.0f, -1.0f / 3.0f,
         1.0f / 6.0f,  SQ2 / 6.0f,  1.0f / 3.0f,
         1.0f / 6.0f, -SQ2 / 6.0f,  1.0f / 3.0f,
         0.0f,         0.0f,        1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs / buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i * 3 + k]
                                   * f[o * channels * 9 + c * 9 + k * 3 + j];
                        }
                        temp[i * 3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi * 3 + k] * G[nu * 3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize
                               + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels + c * outputs + o] =
                        buffer[buffersize * i + entry];
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
        const auto ok =
            phrase_parse(it_line, line.cend(), *x3::float_, x3::space, weights);
        if (!ok || it_line != line.cend()) {
            myprintf("\nFailed to parse weight file. Error on line %d.\n",
                     linecount + 2); //+1 from version line, +1 from 0-indexing
            return {0, 0};
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                m_fwd_weights->m_conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                m_fwd_weights->m_conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                m_fwd_weights->m_batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                m_fwd_weights->m_batchnorm_stddevs.emplace_back(weights);
            }
        } else {
            switch (linecount - plain_conv_wts) {
                case 0: m_fwd_weights->m_conv_pol_w = std::move(weights); break;
                case 1: m_fwd_weights->m_conv_pol_b = std::move(weights); break;
                case 2:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_bn_pol_w1));
                    break;
                case 3:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_bn_pol_w2));
                    break;
                case 4:
                    if (weights.size()
                        != OUTPUTS_POLICY * NUM_INTERSECTIONS
                               * POTENTIAL_MOVES) {
                        myprintf("The weights file is not for %dx%d boards.\n",
                                 BOARD_SIZE, BOARD_SIZE);
                        return {0, 0};
                    }
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip_pol_w));
                    break;
                case 5:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip_pol_b));
                    break;
                case 6: m_fwd_weights->m_conv_val_w = std::move(weights); break;
                case 7: m_fwd_weights->m_conv_val_b = std::move(weights); break;
                case 8:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_bn_val_w1));
                    break;
                case 9:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_bn_val_w2));
                    break;
                case 10:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip1_val_w));
                    break;
                case 11:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip1_val_b));
                    break;
                case 12:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip2_val_w));
                    break;
                case 13:
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip2_val_b));
                    break;
            }
        }
        linecount++;
    }
    process_bn_var(m_bn_pol_w2);
    process_bn_var(m_bn_val_w2);

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

std::unique_ptr<ForwardPipe>&& Network::init_net(
    const int channels, std::unique_ptr<ForwardPipe>&& pipe) {

    pipe->initialize(channels);
    pipe->push_weights(WINOGRAD_ALPHA, INPUT_CHANNELS, channels, m_fwd_weights);

    return std::move(pipe);
}

#ifdef USE_HALF
void Network::select_precision(const int channels) {
    if (cfg_precision == precision_t::AUTO) {
        auto score_fp16 = float{-1.0};
        auto score_fp32 = float{-1.0};

        myprintf("Initializing OpenCL (autodetecting precision).\n");

        // Setup fp16 here so that we can see if we can skip autodetect.
        // However, if fp16 sanity check fails we will return a fp32 and pray it works.
        auto fp16_net = std::make_unique<OpenCLScheduler<half_float::half>>();
        if (!fp16_net->needs_autodetect()) {
            try {
                myprintf("OpenCL: using fp16/half or tensor core compute support.\n");
                m_forward = init_net(channels, std::move(fp16_net));
                benchmark_time(1); // a sanity check run
            } catch (...) {
                myprintf("OpenCL: fp16/half or tensor core failed "
                         "despite driver claiming support.\n");
                myprintf("Falling back to single precision\n");
                m_forward.reset();
                m_forward = init_net(
                    channels, std::make_unique<OpenCLScheduler<float>>());
            }
            return;
        }

        // Start by setting up fp32.
        try {
            m_forward.reset();
            m_forward =
                init_net(channels, std::make_unique<OpenCLScheduler<float>>());
            score_fp32 = benchmark_time(100);
        } catch (...) {
            // empty - if exception thrown just throw away fp32 net
        }

        // Now benchmark fp16.
        try {
            m_forward.reset();
            m_forward = init_net(channels, std::move(fp16_net));
            score_fp16 = benchmark_time(100);
        } catch (...) {
            // empty - if exception thrown just throw away fp16 net
        }

        if (score_fp16 < 0.0f && score_fp32 < 0.0f) {
            myprintf("Both single precision and half precision failed to run.\n");
            throw std::runtime_error("Failed to initialize net.");
        } else if (score_fp16 < 0.0f) {
            myprintf("Using OpenCL single precision (half precision failed to run).\n");
            m_forward.reset();
            m_forward =
                init_net(channels, std::make_unique<OpenCLScheduler<float>>());
        } else if (score_fp32 < 0.0f) {
            myprintf("Using OpenCL half precision (single precision failed to run).\n");
        } else if (score_fp32 * 1.05f > score_fp16) {
            myprintf("Using OpenCL single precision (less than 5%% slower than half).\n");
            m_forward.reset();
            m_forward =
                init_net(channels, std::make_unique<OpenCLScheduler<float>>());
        } else {
            myprintf("Using OpenCL half precision (at least 5%% faster than single).\n");
        }
        return;
    } else if (cfg_precision == precision_t::SINGLE) {
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward =
            init_net(channels, std::make_unique<OpenCLScheduler<float>>());
        return;
    } else if (cfg_precision == precision_t::HALF) {
        myprintf("Initializing OpenCL (half precision).\n");
        m_forward = init_net(
            channels, std::make_unique<OpenCLScheduler<half_float::half>>());
        return;
    }
}
#endif

void Network::initialize(const int playouts, const std::string& weightsfile) {
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    // mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#else
    myprintf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
             EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif

    m_fwd_weights = std::make_shared<ForwardPipeWeights>();

    // Make a guess at a good size as long as the user doesn't
    // explicitly set a maximum memory usage.
    m_nncache.set_size_from_playouts(playouts);

    // Prepare symmetry table
    for (auto s = 0; s < NUM_SYMMETRIES; ++s) {
        for (auto v = 0; v < NUM_INTERSECTIONS; ++v) {
            const auto newvtx =
                get_symmetry({v % BOARD_SIZE, v / BOARD_SIZE}, s);
            symmetry_nn_idx_table[s][v] =
                (newvtx.second * BOARD_SIZE) + newvtx.first;
            assert(symmetry_nn_idx_table[s][v] >= 0
                   && symmetry_nn_idx_table[s][v] < NUM_INTERSECTIONS);
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
    m_fwd_weights->m_conv_weights[weight_index] = winograd_transform_f(
        m_fwd_weights->m_conv_weights[weight_index], channels, INPUT_CHANNELS);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < residual_blocks * 2; i++) {
        m_fwd_weights->m_conv_weights[weight_index] = winograd_transform_f(
            m_fwd_weights->m_conv_weights[weight_index], channels, channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    auto bias_size = m_fwd_weights->m_conv_biases.size();
    for (auto i = size_t{0}; i < bias_size; i++) {
        auto means_size = m_fwd_weights->m_batchnorm_means[i].size();
        for (auto j = size_t{0}; j < means_size; j++) {
            m_fwd_weights->m_batchnorm_means[i][j] -=
                m_fwd_weights->m_conv_biases[i][j];
            m_fwd_weights->m_conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i < m_bn_val_w1.size(); i++) {
        m_bn_val_w1[i] -= m_fwd_weights->m_conv_val_b[i];
        m_fwd_weights->m_conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < m_bn_pol_w1.size(); i++) {
        m_bn_pol_w1[i] -= m_fwd_weights->m_conv_pol_b[i];
        m_fwd_weights->m_conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    if (cfg_cpu_only) {
        myprintf("Initializing CPU-only evaluation.\n");
        m_forward = init_net(channels, std::make_unique<CPUPipe>());
    } else {
#ifdef USE_OPENCL_SELFCHECK
        // initialize CPU reference first, so that we can self-check
        // when doing fp16 vs. fp32 detections
        m_forward_cpu = init_net(channels, std::make_unique<CPUPipe>());
#endif
#ifdef USE_HALF
        // HALF support is enabled, and we are using the GPU.
        // Select the precision to use at runtime.
        select_precision(channels);
#else
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward =
            init_net(channels, std::make_unique<OpenCLScheduler<float>>());
#endif
    }

#else // !USE_OPENCL
    myprintf("Initializing CPU-only evaluation.\n");
    m_forward = init_net(channels, std::make_unique<CPUPipe>());
#endif

    // Need to estimate size before clearing up the pipe.
    get_estimated_size();
    m_fwd_weights.reset();
}

template <unsigned int inputs, unsigned int outputs, bool ReLU, size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases) {
    std::vector<float> output(outputs);

#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);
#else
    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(), inputs, outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
#endif
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = std::max(0.0f, val);
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
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = std::max(0.0f, scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] =
                    std::max(0.0f, (scale_stddiv * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

#ifdef USE_OPENCL_SELFCHECK
void Network::compare_net_outputs(const Netresult& data, const Netresult& ref) {
    // Calculates L2-norm between data and ref.
    constexpr auto max_error = 0.2f;

    auto error = 0.0f;

    for (auto idx = size_t{0}; idx < data.policy.size(); ++idx) {
        const auto diff = data.policy[idx] - ref.policy[idx];
        error += diff * diff;
    }
    const auto diff_pass = data.policy_pass - ref.policy_pass;
    const auto diff_winrate = data.winrate - ref.winrate;
    error += diff_pass * diff_pass;
    error += diff_winrate * diff_winrate;

    error = std::sqrt(error);

    if (error > max_error || std::isnan(error)) {
        printf(
            "Error in OpenCL calculation: Update your device's OpenCL drivers "
            "or reduce the amount of games played simultaneously.\n");
        throw std::runtime_error("OpenCL self-check mismatch.");
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
                for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; ++idx) {
                    const auto sym_idx = symmetry_nn_idx_table[sym][idx];
                    corrected_policy[idx] = result.policy[sym_idx];
                }
                result.policy = std::move(corrected_policy);
                return true;
            }
        }
    }

    return false;
}

Network::Netresult Network::get_output(
    const GameState* const state, const Ensemble ensemble, const int symmetry,
    const bool read_cache, const bool write_cache, const bool force_selfcheck) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (read_cache) {
        // See if we already have this in the cache.
        if (probe_cache(state, result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        assert(symmetry == -1);
        for (auto sym = 0; sym < NUM_SYMMETRIES; ++sym) {
            auto tmpresult = get_output_internal(state, sym);
            result.winrate +=
                tmpresult.winrate / static_cast<float>(NUM_SYMMETRIES);
            result.policy_pass +=
                tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);

            for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
                result.policy[idx] +=
                    tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
#ifdef USE_OPENCL_SELFCHECK
        // Both implementations are available, self-check the OpenCL driver by
        // running both with a probability of 1/2000.
        // selfcheck is done here because this is the only place NN
        // evaluation is done on actual gameplay.
        if (m_forward_cpu != nullptr
            && (force_selfcheck
                || Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0)) {
            auto result_ref = get_output_internal(state, rand_sym, true);
            compare_net_outputs(result, result_ref);
        }
#else
        (void)force_selfcheck;
#endif
    }

    // v2 format (ELF Open Go) returns black value, not stm
    if (m_value_head_not_stm) {
        if (state->board.get_to_move() == FastBoard::WHITE) {
            result.winrate = 1.0f - result.winrate;
        }
    }

    if (write_cache) {
        // Insert result into cache.
        m_nncache.insert(state->board.get_hash(), result);
    }

    return result;
}

Network::Netresult Network::get_output_internal(const GameState* const state,
                                                const int symmetry,
                                                bool selfcheck) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    const auto input_data = gather_features(state, symmetry);
    std::vector<float> policy_data(OUTPUTS_POLICY * width * height);
    std::vector<float> value_data(OUTPUTS_VALUE * width * height);
#ifdef USE_OPENCL_SELFCHECK
    if (selfcheck) {
        m_forward_cpu->forward(input_data, policy_data, value_data);
    } else {
        m_forward->forward(input_data, policy_data, value_data);
    }
#else
    m_forward->forward(input_data, policy_data, value_data);
    (void)selfcheck;
#endif

    // Get the moves
    batchnorm<NUM_INTERSECTIONS>(OUTPUTS_POLICY, policy_data,
                                 m_bn_pol_w1.data(), m_bn_pol_w2.data());
    const auto policy_out =
        innerproduct<OUTPUTS_POLICY * NUM_INTERSECTIONS, POTENTIAL_MOVES,
                     false>(policy_data, m_ip_pol_w, m_ip_pol_b);
    const auto outputs = softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    batchnorm<NUM_INTERSECTIONS>(OUTPUTS_VALUE, value_data, m_bn_val_w1.data(),
                                 m_bn_val_w2.data());
    const auto winrate_data =
        innerproduct<OUTPUTS_VALUE * NUM_INTERSECTIONS, VALUE_LAYER, true>(
            value_data, m_ip1_val_w, m_ip1_val_b);
    const auto winrate_out = innerproduct<VALUE_LAYER, 1, false>(
        winrate_data, m_ip2_val_w, m_ip2_val_b);

    // Map TanH output range [-1..1] to [0..1] range
    const auto winrate = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    Netresult result;

    for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = outputs[idx];
    }

    result.policy_pass = outputs[NUM_INTERSECTIONS];
    result.winrate = winrate;

    return result;
}

void Network::show_heatmap(const FastState* const state,
                           const Netresult& result, const bool topmoves) {
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            auto policy = 0;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
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
        for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
                moves.emplace_back(result.policy[i], vertex);
            }
        }
        moves.emplace_back(result.policy_pass, FastBoard::PASS);

        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        for (const auto& move : moves) {
            if (cum > 0.85f || move.first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                     move.first,
                     state->board.move_to_text(move.second).c_str());
            cum += move.first;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_state(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = float(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = float(true);
        }
    }
}

float Network::get_normalised_komi(const GameState* const state) {
    return 0.5f + (state->get_komi() / (2.0f * TRAINED_UNIT_KOMI));
}

std::vector<float> Network::gather_features(const GameState* const state,
                                            const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    auto input_data = std::vector<float>(INPUT_CHANNELS * NUM_INTERSECTIONS);

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    const auto black_it =
        blacks_move ? begin(input_data)
                    : begin(input_data) + INPUT_MOVES * NUM_INTERSECTIONS;
    const auto white_it =
        blacks_move ? begin(input_data) + INPUT_MOVES * NUM_INTERSECTIONS
                    : begin(input_data);

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_board(h),
                              black_it + h * NUM_INTERSECTIONS,
                              white_it + h * NUM_INTERSECTIONS, symmetry);
    }

    const auto black_to_move_it =
        begin(input_data) + 2 * INPUT_MOVES * NUM_INTERSECTIONS;
    const auto white_to_move_it = black_to_move_it + NUM_INTERSECTIONS;
    const auto pos_norm_komi = get_normalised_komi(state);
    const auto neg_norm_komi = 1.0f - pos_norm_komi;
    const auto black_norm_komi = blacks_move ? pos_norm_komi : neg_norm_komi;
    const auto white_norm_komi = blacks_move ? neg_norm_komi : pos_norm_komi;

    std::fill(black_to_move_it, black_to_move_it + NUM_INTERSECTIONS,
              black_norm_komi);
    std::fill(white_to_move_it, white_to_move_it + NUM_INTERSECTIONS,
              white_norm_komi);

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

size_t Network::get_estimated_size() {
    if (estimated_size != 0) {
        return estimated_size;
    }
    auto result = size_t{0};

    const auto lambda_vector_size =
        [](const std::vector<std::vector<float>>& v) {
            auto result = size_t{0};
            for (auto it = begin(v); it != end(v); ++it) {
                result += it->size() * sizeof(float);
            }
            return result;
        };

    result += lambda_vector_size(m_fwd_weights->m_conv_weights);
    result += lambda_vector_size(m_fwd_weights->m_conv_biases);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_means);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_stddevs);

    result += m_fwd_weights->m_conv_pol_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_pol_b.size() * sizeof(float);

    // Policy head
    result += OUTPUTS_POLICY * sizeof(float); // m_bn_pol_w1
    result += OUTPUTS_POLICY * sizeof(float); // m_bn_pol_w2
    result += OUTPUTS_POLICY * NUM_INTERSECTIONS * POTENTIAL_MOVES
              * sizeof(float);                 // m_ip_pol_w
    result += POTENTIAL_MOVES * sizeof(float); // m_ip_pol_b

    // Value head
    result += m_fwd_weights->m_conv_val_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_val_b.size() * sizeof(float);
    result += OUTPUTS_VALUE * sizeof(float); // m_bn_val_w1
    result += OUTPUTS_VALUE * sizeof(float); // m_bn_val_w2

    result += OUTPUTS_VALUE * NUM_INTERSECTIONS * VALUE_LAYER
              * sizeof(float);             // m_ip1_val_w
    result += VALUE_LAYER * sizeof(float); // m_ip1_val_b

    result += VALUE_LAYER * sizeof(float); // m_ip2_val_w
    result += sizeof(float);               // m_ip2_val_b
    return estimated_size = result;
}

size_t Network::get_estimated_cache_size() {
    return m_nncache.get_estimated_size();
}

void Network::nncache_resize(const int max_count) {
    return m_nncache.resize(max_count);
}

void Network::nncache_clear() {
    m_nncache.clear();
}

void Network::drain_evals() {
    m_forward->drain();
}

void Network::resume_evals() {
    m_forward->resume();
}
