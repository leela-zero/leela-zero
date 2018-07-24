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

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#ifndef USE_BLAS
#error "No non-BLAS implementation"
#endif

#include "CPUPipe.h"
#include "Network.h"
#include "Im2Col.h"

void CPUPipe::initialize(int channels) {
    m_input_channels = channels;
}

void CPUPipe::winograd_transform_in(const std::vector<float>& in,
                                    std::vector<float>& V,
                                    const int C) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    constexpr auto Wpad = 2 + WINOGRAD_M * WTILES;

    std::array<std::array<float, Wpad>, Wpad> in_pad;
    for (auto xin = size_t{0}; xin < Wpad; xin++) {
        for (auto yin = size_t{0}; yin < Wpad; yin++) {
            in_pad[yin][xin] = 0.0f;
        }
    }

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch*(W*H) + yin*W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = WINOGRAD_M * block_x;

                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;
                WinogradTile T1, T2;

                const auto Bt = std::array<float, WINOGRAD_TILE>
                           {1.0f,  0.0f,     -5.0f/2.0f,  0.0f,      1.0f, 0.0f,
                            0.0f, -SQ2,      -2.0f,       SQ2/2.0f,  1.0f, 0.0f,
                            0.0f,  SQ2,      -2.0f,      -SQ2/2.0f,  1.0f, 0.0f,
                            0.0f, -SQ2/2.0f, -1.0f/2.0f,  SQ2,       1.0f, 0.0f,
                            0.0f,  SQ2/2.0f, -1.0f/2.0f, -SQ2,       1.0f, 0.0f,
                            0.0f,  1.0f,      0.0f,      -5.0f/2.0f, 0.0f, 1.0f};

                // Calculates transpose(B).x.B
                for (auto i = 0; i < WINOGRAD_ALPHA; i++){
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < WINOGRAD_ALPHA; k++) {
                            acc += Bt[i * WINOGRAD_ALPHA + k] * \
                                   in_pad[yin + k][xin + j];
                        }
                        T1[i][j] = acc;
                    }
                }

                for (auto i = 0; i < WINOGRAD_ALPHA; i++){
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < WINOGRAD_ALPHA; k++) {
                            acc += T1[i][k] * Bt[j * WINOGRAD_ALPHA + k];
                        }
                        T2[i][j] = acc;
                    }
                }

                const auto offset = ch * P + block_y * WTILES + block_x;
                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        V[(i * WINOGRAD_ALPHA + j)*C*P + offset] = T2[i][j];
                    }
                }
            }
        }
    }
}

void CPUPipe::winograd_sgemm(const std::vector<float>& U,
                             const std::vector<float>& V,
                             std::vector<float>& M,
                             const int C, const int K) {
    constexpr auto P = WINOGRAD_P;

    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * P;
        const auto offset_m = b * K * P;

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
    }
}

void CPUPipe::winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y,
                                     const int K) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    for (auto k = 0; k < K; k++) {
        for (auto block_x = 0; block_x < WTILES; block_x++) {
            const auto x = WINOGRAD_M * block_x;
            for (auto block_y = 0; block_y < WTILES; block_y++) {
                const auto y = WINOGRAD_M * block_y;

                const auto b = block_y * WTILES + block_x;
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;
                WinogradTile temp_m;
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] =
                            M[(xi*WINOGRAD_ALPHA + nu)*K*P + k*P + b];
                    }
                }

                const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>
                      {1.0f, 1.0f,      1.0f,       1.0f,      1.0f,     0.0f,
                       0.0f, SQ2/2.0f, -SQ2/2.0f,   SQ2,      -SQ2,      0.0f,
                       0.0f, 1.0f/2.0f, 1.0f/2.0f,  2.0f,      2.0f,     0.0f,
                       0.0f, SQ2/4.0f, -SQ2/4.0f,   2.0f*SQ2, -2.0f*SQ2, 1.0f};

                std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_M> temp;
                std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M> o;

                // Calculates transpose(A).temp_m.A
                for (auto i = 0; i < WINOGRAD_M; i++){
                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        auto acc = 0.0f;
                        for (auto q = 0; q < WINOGRAD_ALPHA; q++) {
                            acc += At[i * WINOGRAD_ALPHA + q] * temp_m[q][j];
                        }
                        temp[i][j] = acc;
                    }
                }

                for (auto i = 0; i < WINOGRAD_M; i++){
                    for (auto j = 0; j < WINOGRAD_M; j++) {
                        auto acc = 0.0f;
                        for (auto q = 0; q < WINOGRAD_ALPHA; q++) {
                            acc += temp[i][q] * At[j * WINOGRAD_ALPHA + q];
                        }
                        o[i][j] = acc;
                    }
                }

                const auto y_ind = k * H * W + y * W + x;
                for (auto i = 0; i < WINOGRAD_M; i++) {
                    for (auto j = 0; j < WINOGRAD_M; j++) {
                        if (y + i < H && x + j < W) {
                            Y[y_ind + i * W + j] = o[i][j];
                        }
                    }
                }
            }
        }
    }
}

void CPUPipe::winograd_convolve3(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}

template<unsigned int filter_size>
void convolve(const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // The size of the board is defined at compile time
    constexpr unsigned int width = BOARD_SIZE;
    constexpr unsigned int height = BOARD_SIZE;
    constexpr auto board_squares = width * height;
    constexpr auto filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, board_squares, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], board_squares,
                0.0f, &output[0], board_squares);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < board_squares; b++) {
            output[(o * board_squares) + b] += biases[o];
        }
    }
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

void CPUPipe::forward(const std::vector<float>& input,
                      std::vector<float>& output_pol,
                      std::vector<float>& output_val) {
    // Input convolution
    constexpr auto P = WINOGRAD_P;
    // Calculate output channels
    const auto output_channels = m_input_channels;
    // input_channels is the maximum number of input channels of any
    // convolution. Residual blocks are identical, but the first convolution
    // might be bigger when the network has very few filters
    const auto input_channels = std::max(static_cast<size_t>(output_channels),
                                         static_cast<size_t>(Network::INPUT_CHANNELS));
    auto conv_out = std::vector<float>(output_channels * BOARD_SQUARES);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * P);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * P);

    winograd_convolve3(output_channels, input, m_conv_weights[0], V, M, conv_out);
    batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                             m_batchnorm_means[0].data(),
                             m_batchnorm_stddivs[0].data());

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * BOARD_SQUARES);
    auto res = std::vector<float>(output_channels * BOARD_SQUARES);
    for (auto i = size_t{1}; i < m_conv_weights.size(); i += 2) {
        auto output_channels = m_input_channels;
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           m_conv_weights[i], V, M, conv_out);
        batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                                 m_batchnorm_means[i].data(),
                                 m_batchnorm_stddivs[i].data());

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           m_conv_weights[i + 1], V, M, conv_out);
        batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                                 m_batchnorm_means[i + 1].data(),
                                 m_batchnorm_stddivs[i + 1].data(),
                                 res.data());
    }
    convolve<1>(Network::OUTPUTS_POLICY, conv_out, m_conv_pol_w, m_conv_pol_b, output_pol);
    convolve<1>(Network::OUTPUTS_VALUE, conv_out, m_conv_val_w, m_conv_val_b, output_val);
}


void CPUPipe::push_input_convolution(unsigned int /*filter_size*/,
                                     unsigned int /*channels*/,
                                     unsigned int /*outputs*/,
                                     const std::vector<float>& weights,
                                     const std::vector<float>& means,
                                     const std::vector<float>& variances) {
    m_conv_weights.push_back(weights);
    m_batchnorm_means.push_back(means);
    m_batchnorm_stddivs.push_back(variances);
}

void CPUPipe::push_residual(unsigned int /*filter_size*/,
                            unsigned int /*channels*/,
                            unsigned int /*outputs*/,
                            const std::vector<float>& weights_1,
                            const std::vector<float>& means_1,
                            const std::vector<float>& variances_1,
                            const std::vector<float>& weights_2,
                            const std::vector<float>& means_2,
                            const std::vector<float>& variances_2) {
    m_conv_weights.push_back(weights_1);
    m_batchnorm_means.push_back(means_1);
    m_batchnorm_stddivs.push_back(variances_1);

    m_conv_weights.push_back(weights_2);
    m_batchnorm_means.push_back(means_2);
    m_batchnorm_stddivs.push_back(variances_2);
}

void CPUPipe::push_convolve(unsigned int filter_size,
                            unsigned int channels,
                            unsigned int outputs,
                            const std::vector<float>& weights) {
    // currently we can only support the final convolve stages
    (void)filter_size;
    assert(filter_size == 1);

    if (outputs == Network::OUTPUTS_POLICY) {
        m_conv_pol_w = weights;
        m_conv_pol_b.resize(m_conv_pol_w.size() / channels, 0.0f);
    } else if (outputs == Network::OUTPUTS_VALUE) {
        m_conv_val_w = weights;
        m_conv_val_b.resize(m_conv_val_w.size() / channels, 0.0f);
    } else {
        assert(false);
    }
}
