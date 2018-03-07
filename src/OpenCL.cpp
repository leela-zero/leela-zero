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

#ifdef USE_OPENCL
#include "OpenCL.h"

#include <cassert>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iterator>
#include <limits>
#include <stdexcept>

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "Network.h"
#include "GTP.h"
#include "Utils.h"
#include "Tuner.h"

using namespace Utils;

static std::string cl_args =
    "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";

static std::string sourceCode_config = R"(
    typedef float net_t;
    #define vload_net_t(offset,p) ((p)[(offset)])
    #define vstore_net_t(data,offset,p) (((p)[(offset)])=(data))
    #define BOARD_SIZE )" + std::to_string(BOARD_SIZE) +
    "\n    #define BOARD_SQUARES " + std::to_string(BOARD_SQUARES);

static std::string sourceCode_convolve1 = R"(
    __kernel
    __attribute__((work_group_size_hint(8, 16, 1)))
    void convolve1(
                   __global const net_t * restrict in,
                   __global net_t * restrict merge,
                   __global const net_t * restrict weights,
                   __local float * channel_buff,
                   __local float * row_buff) {
        // cl::NDRange global(channels, outputs, row);
        const int c   = get_global_id(0);  // channel
        const int o   = get_global_id(1);  // output
        const int row = get_global_id(2);  // row
        const int channels = get_global_size(0);
        const int outputs  = get_global_size(1);
        // cl::NDRange local(2, (1->32), 1);
        const int lx = get_local_id(0);
        const int ly = get_local_id(1);
        const int chan_buff_size = 8;
        const int out_buff_size  = get_local_size(1);
        const int row_buff_size  = 7;
        const int chan_shift     = 3;
        // input = channels * height * width
        // output = outputs * height * width
        // weights = output * channels * filter
        // merge = channels * outputs * height * width
        const int width = BOARD_SIZE;
        const int height = BOARD_SIZE;
        const int strip_size = width;
        // Copy the input channels (strips) locally
        if (out_buff_size < BOARD_SIZE && ly == 0) {
            // strip-row
            for (int w = 0; w < width; w++) {
                channel_buff[lx * width + w] =
                    vload_net_t((c * height + row) * width + w, in);
            }
        } else if (out_buff_size >= BOARD_SIZE && ly < BOARD_SIZE) {
            // Every thread copies a column
            channel_buff[lx * width + ly] = vload_net_t((c * height + row) * width + ly, in);
        }
        // Copy the filter we are applying locally
        __private float filter_buff = vload_net_t((o * channels + c), weights);
        barrier(CLK_LOCAL_MEM_FENCE);
        int out_lane = 0;
        int out_cw   = 0;
        #pragma unroll
        for (int cw = 0; cw < width; cw++) {
            int fid = lx * strip_size;
            float out  = channel_buff[fid + cw] * filter_buff;
            row_buff[(ly * chan_buff_size + lx) * row_buff_size + out_lane] = out;
            out_lane++;
            // Row buffer full or last lane?
            if (out_lane == row_buff_size || (cw == width - 1)) {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (lx < out_lane) {
                    float val;
                    val  = row_buff[(ly * chan_buff_size + 0) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 1) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 2) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 3) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 4) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 5) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 6) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 7) * row_buff_size + lx];
                    vstore_net_t(val, (((c >> chan_shift) * height + row) * width + out_cw + lx) * outputs + o, merge);
                }
                out_cw  += row_buff_size;
                out_lane = 0;
           }
       }
    }

__kernel void merge(
                        __global const net_t * restrict in,
                        __global net_t * restrict out,
                        __private const int channels) {
        // cl::NDRange global(outputs, BOARD_SQUARES);
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const int output = gx;
        const int b = gy;
        const int outputs = get_global_size(0);
        const int width = BOARD_SIZE;
        const int height = BOARD_SIZE;
        const int o = output;
        float sum = 0;
        for (int c = 0; c < channels; c++) {
            sum += vload_net_t((c * BOARD_SQUARES + b) * outputs + o, in);
        }
        vstore_net_t(sum, o * BOARD_SQUARES + b, out);
    }
)";

static std::string sourceCode_convolve3 = R"(
void __in_transform_eq(float x[4][4], __global float * restrict V, int offset, int CPpad) {
    float T1[4][4];

    T1[0][0] = x[0][0] - x[2][0];
    T1[0][1] = x[0][1] - x[2][1];
    T1[0][2] = x[0][2] - x[2][2];
    T1[0][3] = x[0][3] - x[2][3];
    T1[1][0] = x[1][0] + x[2][0];
    T1[1][1] = x[1][1] + x[2][1];
    T1[1][2] = x[1][2] + x[2][2];
    T1[1][3] = x[1][3] + x[2][3];
    T1[2][0] = x[2][0] - x[1][0];
    T1[2][1] = x[2][1] - x[1][1];
    T1[2][2] = x[2][2] - x[1][2];
    T1[2][3] = x[2][3] - x[1][3];
    T1[3][0] = x[1][0] - x[3][0];
    T1[3][1] = x[1][1] - x[3][1];
    T1[3][2] = x[1][2] - x[3][2];
    T1[3][3] = x[1][3] - x[3][3];

    V[(0*4 + 0)*CPpad + offset] = T1[0][0] - T1[0][2];
    V[(0*4 + 1)*CPpad + offset] = T1[0][1] + T1[0][2];
    V[(0*4 + 2)*CPpad + offset] = T1[0][2] - T1[0][1];
    V[(0*4 + 3)*CPpad + offset] = T1[0][1] - T1[0][3];
    V[(1*4 + 0)*CPpad + offset] = T1[1][0] - T1[1][2];
    V[(1*4 + 1)*CPpad + offset] = T1[1][1] + T1[1][2];
    V[(1*4 + 2)*CPpad + offset] = T1[1][2] - T1[1][1];
    V[(1*4 + 3)*CPpad + offset] = T1[1][1] - T1[1][3];
    V[(2*4 + 0)*CPpad + offset] = T1[2][0] - T1[2][2];
    V[(2*4 + 1)*CPpad + offset] = T1[2][1] + T1[2][2];
    V[(2*4 + 2)*CPpad + offset] = T1[2][2] - T1[2][1];
    V[(2*4 + 3)*CPpad + offset] = T1[2][1] - T1[2][3];
    V[(3*4 + 0)*CPpad + offset] = T1[3][0] - T1[3][2];
    V[(3*4 + 1)*CPpad + offset] = T1[3][1] + T1[3][2];
    V[(3*4 + 2)*CPpad + offset] = T1[3][2] - T1[3][1];
    V[(3*4 + 3)*CPpad + offset] = T1[3][1] - T1[3][3];
}

__kernel void in_transform(__global net_t * restrict in, __global float * restrict V,
                           const int C, const int Cpad,
                           const int Ppad) {
    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int T = W*H;
    const int WTILES = (W + 1) / 2;
    const int P = WTILES*WTILES;
    const int CPpad = Ppad * Cpad;

    const int block = get_global_id(0);
    const int ch = get_global_id(1);
    const int chT = ch*(T);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    // Tiles overlap by 2
    const int yin = 2 * block_y - 1;
    const int xin = 2 * block_x - 1;

    if (block < P && ch < C) {
        // Cache input tile and handle zero padding
        float x[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int a = xin + j;
                int b = yin + i;
                if (b >= 0 && a >= 0 && b < H && a < W) {
                    x[i][j] = vload_net_t(chT + b*W + a, in);
                } else {
                    x[i][j] = 0.0f;
                }
            }
        }

        const int offset = ch*Ppad + block;
        __in_transform_eq(x, V, offset, CPpad);
    }
}

void __out_transform_eq(__global const float * restrict M, float o[4],
                        int Kpad, int Ppad, int block_x, int block_y)
{
    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int WTILES = (W + 1) / 2;
    const int b = block_y * WTILES + block_x;
    const int KPpad = Kpad * Ppad;
    const int k = get_global_id(0);
    float temp_m[16];
    for (int xn = 0, xnKPpad = b*Kpad + k; xn < 16; xn++, xnKPpad += KPpad) {
        temp_m[xn] = M[xnKPpad];
    }

    o[0] = temp_m[0*4 + 0] + temp_m[0*4 + 1] + temp_m[0*4 + 2] +
           temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] +
           temp_m[2*4 + 0] + temp_m[2*4 + 1] + temp_m[2*4 + 2];

    o[1] = temp_m[0*4 + 1] - temp_m[0*4 + 2] - temp_m[0*4 + 3] +
           temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] +
           temp_m[2*4 + 1] - temp_m[2*4 + 2] - temp_m[2*4 + 3];

    o[2] = temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] -
           temp_m[2*4 + 0] - temp_m[2*4 + 1] - temp_m[2*4 + 2] -
           temp_m[3*4 + 0] - temp_m[3*4 + 1] - temp_m[3*4 + 2];

    o[3] = temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] -
           temp_m[2*4 + 1] + temp_m[2*4 + 2] + temp_m[2*4 + 3] -
           temp_m[3*4 + 1] + temp_m[3*4 + 2] + temp_m[3*4 + 3];
}

__kernel void out_transform_fused_bn(__global const float * restrict M,
                                     __global net_t * restrict Y,
                                     const int K,
                                     const int Kpad, const int Ppad,
                                     __global const net_t * restrict residual,
                                     __constant const net_t * restrict means,
                                     __constant const net_t * restrict stddivs) {
    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int WTILES = (W + 1) / 2;
    const int P = WTILES * WTILES;

    int k = get_global_id(0);
    int block = get_global_id(1);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    int x = 2*block_x;
    int y = 2*block_y;
    int a_ind = (y)*W + (x);
    if (k < K && block < P) {
        const int kHW = k * W * H;
        float o[4];
        __out_transform_eq(M, o, Kpad, Ppad, block_x, block_y);

        const float mean = vload_net_t(k, means);
        const float scale_stddiv = vload_net_t(k, stddivs);

        const bool pred[4] = { 1, x+1 < W, y+1 < H, x+1 < W & y+1 < H};

        const int a[4] = {a_ind, a_ind+1, a_ind+W, a_ind+W+1};

        for (int i = 0; i < 4; i++) {
            if (pred[i]) {
                o[i] = scale_stddiv * (o[i] - mean);
                if (residual) {
                    o[i] += vload_net_t(kHW + a[i], residual);
                }
                o[i] = o[i] > 0 ? o[i] : 0.0f;
                vstore_net_t(o[i], kHW + a[i], Y);
            }
        }
    }
}

__kernel void out_transform_fused_bn_in(
                                     __global const float * restrict M,
                                     __global net_t * restrict Y,
                                     __global net_t * restrict V,
                                     const int K,
                                     const int Kpad, const int Ppad, const int Cpad,
                                     __global const net_t * restrict residual,
                                     __constant const net_t * restrict means,
                                     __constant const net_t * restrict stddivs,
                                     __local float * ybuf) {
    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int T = W*H;
    const int WTILES = (W + 1) / 2;
    const int P = WTILES * WTILES;
    const int KPpad = Kpad * Ppad;

    const int k = get_global_id(0);
    const int kg = get_local_id(0);
    const int block = get_global_id(1);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    const int yin = 2 * block_y - 1;
    const int xin = 2 * block_x - 1;


    const int x = 2*block_x;
    const int y = 2*block_y;
    int a_ind = (y)*W + (x);


    if (k < K && block < P) {
        const int a[4] = {a_ind, a_ind+1, a_ind+W, a_ind+W+1};
        const bool pred[4] = { 1, x+1 < W, y+1 < H, x+1 < W & y+1 < H};
        const int kHW = k * W * H;

        float o[4];
        __out_transform_eq(M, o, Kpad, Ppad, block_x, block_y);

        const float mean = vload_net_t(k, means);
        const float scale_stddiv = vload_net_t(k, stddivs);

        for (int i = 0; i < 4; i++) {
            if (pred[i]) {
                o[i] = scale_stddiv * (o[i] - mean);
                if (residual) {
                    o[i] += vload_net_t(kHW + a[i], residual);
                }
                o[i] = o[i] > 0 ? o[i] : 0.0f;
                ybuf[kg * T + a[i]] = o[i];
                if (Y) {
                    vstore_net_t(o[i], kHW + a[i], Y);
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (block < P && k < K) {
        const int CPpad = Ppad * Cpad;
        // Cache input tile and handle zero padding
        float xx[4][4];
        for (int i = 0; i < 4; i++) {
            int b = yin + i;
            for (int j = 0; j < 4; j++) {
                int a = xin + j;
                if (b >= 0 && a >= 0 && b < H && a < W) {
                    xx[i][j] = ybuf[kg * T + b*W + a];
                } else {
                    xx[i][j] = 0.0f;
                }
            }
        }

        const int offset = k*Ppad + block;
        __in_transform_eq(xx, V, offset, CPpad);
    }
}
)";

const std::string sourceCode_sgemm =
    #include "clblast_level3/common.opencl"
    #include "clblast_level3/xgemm_part1.opencl"
    #include "clblast_level3/xgemm_part2.opencl"
    #include "clblast_level3/xgemm_part3.opencl"
    #include "clblast_level3/xgemm_batched.opencl"
;

thread_local ThreadData opencl_thread_data;

void OpenCL::ensure_thread_initialized() {
    if (!opencl_thread_data.m_is_initialized) {
        // Make kernels
        opencl_thread_data.m_convolve1_kernel =
            cl::Kernel(m_program, "convolve1");
        opencl_thread_data.m_merge_kernel =
            cl::Kernel(m_program, "merge");
        opencl_thread_data.m_in_transform_kernel =
            cl::Kernel(m_program, "in_transform");
        opencl_thread_data.m_sgemm_kernel =
            cl::Kernel(m_program, "XgemmBatched");
        opencl_thread_data.m_out_transform_bn_kernel =
            cl::Kernel(m_program, "out_transform_fused_bn");
        opencl_thread_data.m_out_transform_bn_in_kernel =
            cl::Kernel(m_program, "out_transform_fused_bn_in");
        opencl_thread_data.m_commandqueue =
            cl::CommandQueue(m_context, m_device);
        opencl_thread_data.m_is_initialized = true;
    }
}

void OpenCL_Network::add_weights(size_t layer,
                                 size_t size,
                                 const float * weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(Layer());
    }

    auto converted_weights = std::vector<net_t>();
    for(auto i = size_t{0}; i < size; i++) {
        converted_weights.emplace_back(weights[i]);
    }

    auto weightSize = size * sizeof(decltype(converted_weights)::value_type);
    m_layers.back().weights.emplace_back(
        m_opencl.m_context,
        CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
        weightSize,
        const_cast<net_t*>(converted_weights.data()));
}

void OpenCL_Network::forward(const std::vector<net_t>& input,
                             std::vector<net_t>& output_pol,
                             std::vector<net_t>& output_val) {
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;
    constexpr auto tiles = WINOGRAD_P;
    constexpr auto one_plane = width * height * sizeof(net_t);
    const auto finalSize_pol = m_layers[m_layers.size()-2].outputs * one_plane;
    const auto finalSize_val = m_layers.back().outputs * one_plane;

    m_opencl.ensure_thread_initialized();

    if (!opencl_thread_data.m_buffers_allocated) {
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_channels = std::max(max_channels,
                                    std::max(layer.channels, layer.outputs));
        }

        const auto mwg = m_opencl.m_sgemm_tuners.mwg;
        const auto nwg = m_opencl.m_sgemm_tuners.nwg;
        const auto vwm = m_opencl.m_sgemm_tuners.vwm;
        const auto vwn = m_opencl.m_sgemm_tuners.vwn;

        const auto m_ceil = ceilMultiple(ceilMultiple(max_channels, mwg), vwm);
        const auto n_ceil = ceilMultiple(ceilMultiple(tiles, nwg), vwn);

        const auto alloc_inSize =
            m_ceil * m_ceil *  max_channels * sizeof(net_t);
        const auto alloc_vm_size =
            WINOGRAD_TILE * m_ceil * n_ceil * sizeof(net_t);

        auto v_zeros = std::vector<float>(alloc_vm_size);

        opencl_thread_data.m_inBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_inBuffer2 = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_VBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
            alloc_vm_size, v_zeros.data(), nullptr);
        opencl_thread_data.m_MBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_vm_size);

        opencl_thread_data.m_pinnedOutBuffer_pol = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, finalSize_pol);
        opencl_thread_data.m_pinnedOutBuffer_val = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, finalSize_val);

        opencl_thread_data.m_buffers_allocated = true;
    }

    cl::Buffer & inBuffer = opencl_thread_data.m_inBuffer;
    cl::Buffer & inBuffer2 = opencl_thread_data.m_inBuffer2;
    cl::Buffer & VBuffer = opencl_thread_data.m_VBuffer;
    cl::Buffer & MBuffer = opencl_thread_data.m_MBuffer;
    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    const auto inSize = sizeof(net_t) * input.size();
    queue.enqueueWriteBuffer(inBuffer, CL_FALSE, 0, inSize, input.data());

    auto skip_in_trans = false;
    for (auto iter = cbegin(m_layers); iter != cend(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            assert(niter != cend(m_layers));
            auto conv_weights = begin(layer.weights);
            auto bn_weights = begin(layer.weights) + 1;
            auto skip_next_in_trans = false;
            if (niter->is_residual_block) {
                skip_next_in_trans = true;
            }
            convolve3(layer.channels,
                     layer.outputs,
                     inBuffer,
                     inBuffer,
                     VBuffer,
                     MBuffer,
                     conv_weights,
                     nullptr,
                     bn_weights,
                     skip_in_trans, skip_next_in_trans, true);
            skip_in_trans = skip_next_in_trans;
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != cend(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto bn1_weights   = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 3;
            auto bn2_weights   = begin(layer.weights) + 4;
            convolve3(layer.channels,
                      layer.outputs,
                      inBuffer,
                      inBuffer2,
                      VBuffer,
                      MBuffer,
                      conv1_weights,
                      nullptr,
                      bn1_weights,
                      skip_in_trans, true, false);

            auto skip_next_in_trans = false;
            if (niter->is_residual_block) {
                skip_next_in_trans = true;
            }
            convolve3(layer.channels,
                      layer.outputs,
                      inBuffer2,
                      inBuffer,
                      VBuffer,
                      MBuffer,
                      conv2_weights,
                      &inBuffer,
                      bn2_weights,
                      true, skip_next_in_trans, true);
            skip_in_trans = skip_next_in_trans;
        } else {
            assert(layer.is_convolve1);

            cl::Buffer out_buffer;
            if (niter == cend(m_layers)) {
                out_buffer = opencl_thread_data.m_pinnedOutBuffer_val;
            } else {
                out_buffer = opencl_thread_data.m_pinnedOutBuffer_pol;
            }

            convolve1(layer.channels,
                    layer.outputs,
                    inBuffer,
                    out_buffer,
                    VBuffer,
                    begin(layer.weights));
        }
    }

    auto pinnedOutBufferHost_pol = queue.enqueueMapBuffer(
        opencl_thread_data.m_pinnedOutBuffer_pol, CL_FALSE,
        CL_MAP_READ, 0, finalSize_pol);
    auto pinnedOutBufferHost_val = queue.enqueueMapBuffer(
        opencl_thread_data.m_pinnedOutBuffer_val, CL_FALSE,
        CL_MAP_READ, 0, finalSize_val);

    {
        // Finish call is usually a busy wait. When using multiple threads
        // use the lock to avoid busy waiting with all threads.
        std::lock_guard<std::mutex> lock(m_queue_finish_mutex);
        queue.finish();
    }

    std::memcpy(output_pol.data(), pinnedOutBufferHost_pol, finalSize_pol);
    std::memcpy(output_val.data(), pinnedOutBufferHost_val, finalSize_val);

    queue.enqueueUnmapMemObject(opencl_thread_data.m_pinnedOutBuffer_pol,
            pinnedOutBufferHost_pol);
    queue.enqueueUnmapMemObject(opencl_thread_data.m_pinnedOutBuffer_val,
            pinnedOutBufferHost_val);

}

void OpenCL_Network::convolve3(int channels, int outputs,
                              cl::Buffer& bufferIn,
                              cl::Buffer& bufferOut,
                              cl::Buffer& bufferV,
                              cl::Buffer& bufferM,
                              weight_slice_t weights,
                              cl::Buffer* bufferResidual,
                              weight_slice_t bn_weights,
                              bool skip_in_transform,
                              bool fuse_in_transform,
                              bool store_inout) {

    cl::Kernel & in_transform_kernel = opencl_thread_data.m_in_transform_kernel;
    cl::Kernel & sgemm_kernel = opencl_thread_data.m_sgemm_kernel;
    cl::Kernel & out_transform_bn_kernel =
        opencl_thread_data.m_out_transform_bn_kernel;
    cl::Kernel & out_transform_bn_in_kernel =
        opencl_thread_data.m_out_transform_bn_in_kernel;

    auto mwg = m_opencl.m_sgemm_tuners.mwg;
    auto nwg = m_opencl.m_sgemm_tuners.nwg;
    auto kwg = m_opencl.m_sgemm_tuners.kwg;
    auto vwm = m_opencl.m_sgemm_tuners.vwm;
    auto vwn = m_opencl.m_sgemm_tuners.vwn;
    auto mdimc = m_opencl.m_sgemm_tuners.mdimc;
    auto ndimc = m_opencl.m_sgemm_tuners.ndimc;
    auto wavefront_size = m_opencl.m_wavefront_size;

    assert(mwg != 0);
    assert(nwg != 0);
    assert(kwg != 0);
    assert(mdimc != 0);
    assert(ndimc != 0);
    assert(vwm != 0);
    assert(vwn != 0);
    assert(wavefront_size != 0);

    constexpr auto tiles = WINOGRAD_P;
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    auto wgs = ceilMultiple(tiles, wavefront_size);
    auto m_ceil = int(ceilMultiple(ceilMultiple(outputs, mwg), vwm));
    auto n_ceil = int(ceilMultiple(ceilMultiple(tiles, nwg), vwn));
    auto k_ceil = int(ceilMultiple(ceilMultiple(channels, kwg), vwm));

    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    if (!skip_in_transform) {
        try {
            in_transform_kernel.setArg(0, bufferIn);
            in_transform_kernel.setArg(1, bufferV);
            in_transform_kernel.setArg(2, channels);
            in_transform_kernel.setArg(3, k_ceil);
            in_transform_kernel.setArg(4, n_ceil);

            queue.enqueueNDRangeKernel(in_transform_kernel, cl::NullRange,
                                       cl::NDRange(wgs, channels));
        } catch (const cl::Error &e) {
            std::cerr << "Error in convolve3: " << e.what() << ": "
                << e.err() << std::endl;
            throw;
        }
    }

    try {
        sgemm_kernel.setArg(0, m_ceil);
        sgemm_kernel.setArg(1, n_ceil);
        sgemm_kernel.setArg(2, k_ceil);
        sgemm_kernel.setArg(3, weights[0]);
        sgemm_kernel.setArg(4, bufferV);
        sgemm_kernel.setArg(5, bufferM);

        cl::NDRange local_sgemm = {mdimc, ndimc, 1};

        cl::NDRange size_sgemm = {(m_ceil * mdimc) / mwg,
                                  (n_ceil * ndimc) / nwg,
                                  (cl::size_type)WINOGRAD_TILE};

        queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange,
                                   size_sgemm, local_sgemm);
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3: " << e.what() << ": "
            << e.err() << std::endl;
        throw;
    }

    try {
        if (fuse_in_transform) {
            // TODO : Eventually this might also be something tuneable?
            constexpr auto dim_size = 2;
            out_transform_bn_in_kernel.setArg(0, bufferM);
            if (store_inout) {
                out_transform_bn_in_kernel.setArg(1, bufferOut);
            } else {
                out_transform_bn_in_kernel.setArg(1, nullptr);
            }
            out_transform_bn_in_kernel.setArg(2, bufferV);
            out_transform_bn_in_kernel.setArg(3, outputs);
            out_transform_bn_in_kernel.setArg(4, m_ceil);
            out_transform_bn_in_kernel.setArg(5, n_ceil);
            // k_ceil of the next convolution
            auto k_ceil2 = int(ceilMultiple(ceilMultiple(outputs, kwg), vwm));
            out_transform_bn_in_kernel.setArg(6, k_ceil2);
            if (bufferResidual) {
                out_transform_bn_in_kernel.setArg(7, *bufferResidual);
            } else {
                out_transform_bn_in_kernel.setArg(7, nullptr);
            }
            out_transform_bn_in_kernel.setArg(8, bn_weights[0]);
            out_transform_bn_in_kernel.setArg(9, bn_weights[1]);
            out_transform_bn_in_kernel.setArg(10,
                cl::Local(dim_size * width * height * sizeof(float)));

            queue.enqueueNDRangeKernel(out_transform_bn_in_kernel,
                                       cl::NullRange,
                                       cl::NDRange(outputs, wgs),
                                       cl::NDRange(dim_size, wgs));
        } else {
            out_transform_bn_kernel.setArg(0, bufferM);
            out_transform_bn_kernel.setArg(1, bufferOut);
            out_transform_bn_kernel.setArg(2, outputs);
            out_transform_bn_kernel.setArg(3, m_ceil);
            out_transform_bn_kernel.setArg(4, n_ceil);
            if (bufferResidual) {
                out_transform_bn_kernel.setArg(5, *bufferResidual);
            } else {
                out_transform_bn_kernel.setArg(5, nullptr);
            }
            out_transform_bn_kernel.setArg(6, bn_weights[0]);
            out_transform_bn_kernel.setArg(7, bn_weights[1]);

            queue.enqueueNDRangeKernel(out_transform_bn_kernel, cl::NullRange,
                                       cl::NDRange(outputs, wgs));
        }
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3: " << e.what() << ": "
            << e.err() << std::endl;
        throw;
    }
}

void OpenCL_Network::convolve1(int channels, int outputs,
                              cl::Buffer& bufferInput,
                              cl::Buffer& bufferOutput,
                              cl::Buffer& bufferMerge,
                              weight_slice_t weights) {
    // The size of the board is defined at compile time
    constexpr int width = BOARD_SIZE;
    constexpr int boardsize = BOARD_SQUARES;
    constexpr int rowTiles = BOARD_SIZE;

    // Input channel grouping in multiples of 8
    constexpr int channelGroup = 8;
    constexpr int channelShift = 3;
    constexpr int rowGroup = 1;
    size_t outputGroup = std::min(outputs, 32);

    auto m_convolve_kernel = &opencl_thread_data.m_convolve1_kernel;

#ifndef NDEBUG
    // Total output size after reducing
    size_t outSize = boardsize * outputs * sizeof(net_t);

    // Produce channel * output planes and merge them at the end
    size_t mergeSize = (channels >> channelShift) * outSize;
    assert(mergeSize <= bufferMerge.getInfo<CL_MEM_SIZE>());
#endif

    // Copy the rows locally
    size_t stripSize = width * sizeof(float);

    int rowBuffer = std::min<int>(channelGroup, 7);
    size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);

    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    try {
        m_convolve_kernel->setArg(0, bufferInput);
        m_convolve_kernel->setArg(1, bufferMerge);
        m_convolve_kernel->setArg(2, weights[0]);
        m_convolve_kernel->setArg(3, cl::Local(stripSize * channelGroup * rowGroup));
        m_convolve_kernel->setArg(4, cl::Local(rowSize));

        queue.enqueueNDRangeKernel(*m_convolve_kernel, cl::NullRange,
                                   cl::NDRange(channels, outputs, rowTiles),
                                   cl::NDRange(channelGroup, outputGroup, rowGroup));
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve1: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }

    cl::Kernel & merge_kernel = opencl_thread_data.m_merge_kernel;
    assert(channels % (1 << channelShift) == 0);

    try {
        merge_kernel.setArg(0, bufferMerge);
        merge_kernel.setArg(1, bufferOutput);
        merge_kernel.setArg(2, channels >> channelShift);

        queue.enqueueNDRangeKernel(merge_kernel, cl::NullRange,
                                   cl::NDRange(outputs, boardsize),
                                   cl::NDRange(std::min(8, outputs), BOARD_SIZE));
    } catch (const cl::Error &e) {
        std::cerr << "Error in merge: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }
}

template<class T>
static std::string opencl_dev_type_to_string(T type) {
    if (type == CL_DEVICE_TYPE_CPU) {
        return "CPU";
    } else if (type == CL_DEVICE_TYPE_GPU) {
        return "GPU";
    } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
        return "Accelerator";
    } else {
        return "Unknown";
    }
}

static std::string trim(std::string trim_me) {
    boost::algorithm::trim(trim_me);
    return trim_me;
}

void OpenCL::process_tuners(std::string tuners) {
    std::string buf;
    std::stringstream ss(tuners);
    std::size_t found;

    auto mwg = false;
    auto nwg = false;
    auto kwg = false;
    auto ndimc = false;
    auto mdimc = false;
    auto vwm = false;
    auto vwn = false;
    while (ss >> buf) {
        found = buf.find("=");
        if (found == std::string::npos) {
            std::cerr << "Invalid tuner string: " << tuners << std::endl;
            std::exit(-1);
        }
        std::string name = buf.substr(0, found);
        auto value = std::stoi(buf.substr(found + 1, std::string::npos));
        if (name == "-DMWG") {
            m_sgemm_tuners.mwg = value;
            mwg = true;
        }
        if (name == "-DNWG") {
            m_sgemm_tuners.nwg = value;
            nwg = true;
        }
        if (name == "-DKWG") {
            m_sgemm_tuners.kwg = value;
            kwg = true;
        }
        if (name == "-DMDIMC") {
            m_sgemm_tuners.mdimc = value;
            mdimc = true;
        }
        if (name == "-DNDIMC") {
            m_sgemm_tuners.ndimc = value;
            ndimc = true;
        }
        if (name == "-DVWM") {
            m_sgemm_tuners.vwm = value;
            vwm = true;
        }
        if (name == "-DVWN") {
            m_sgemm_tuners.vwn = value;
            vwn = true;
        }
    }
    if (!mwg || !nwg || !kwg || !mdimc || !ndimc || !vwm || !vwn) {
        std::cerr << "Missing tuner parameters";
        if (!mwg) {
            std::cerr << " MWG";
        }
        if (!nwg) {
            std::cerr << " NWG";
        }
        if (!kwg) {
            std::cerr << " KWG";
        }
        if (!mdimc) {
            std::cerr << " MDIMC";
        }
        if (!ndimc) {
            std::cerr << " NDIMC";
        }
        if (!vwm) {
            std::cerr << " VWM";
        }
        if (!vwn) {
            std::cerr << " VWN";
        }
        std::cerr << std::endl;
        std::exit(-1);
    }
}

std::vector<size_t> OpenCL::get_sgemm_tuners(void) {
    std::vector<size_t> tuners;

    tuners.emplace_back(m_sgemm_tuners.mwg);
    tuners.emplace_back(m_sgemm_tuners.nwg);
    tuners.emplace_back(m_sgemm_tuners.kwg);
    tuners.emplace_back(m_sgemm_tuners.vwm);
    tuners.emplace_back(m_sgemm_tuners.vwn);
    tuners.emplace_back(m_sgemm_tuners.mdimc);
    tuners.emplace_back(m_sgemm_tuners.ndimc);

    return tuners;
}

void OpenCL::initialize(const int channels, const std::vector<int> & gpus,
                        bool silent) {
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error &e) {
        myprintf("OpenCL: %s\n", e.what());
        throw;
    }

    auto best_version = 0.0f;
    cl::Platform best_platform;
    cl::Device best_device;
    std::string best_vendor;
    auto best_score = 0;
    auto found_device = false;
    auto id = 0;

    if (!silent) {
        myprintf("Detected %d OpenCL platforms.\n", platforms.size());
    }

    for (const auto &p : platforms) {
        std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
        if (!silent) {
            std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
            std::string platname = p.getInfo<CL_PLATFORM_NAME>();
            std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
            myprintf("Platform version: %s\n", platvers.c_str());;
            myprintf("Platform profile: %s\n", platprof.c_str());
            myprintf("Platform name:    %s\n", platname.c_str());
            myprintf("Platform vendor:  %s\n", platvend.c_str());
        }

        std::istringstream versstream(platvers);
        std::string tmp;
        float opencl_version;
        versstream >> tmp >> opencl_version;

        std::vector<cl::Device> devices;
        try {
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        } catch (const cl::Error &e) {
            myprintf("Error getting device(s): %s: %d\n", e.what(), e.err());
            devices.clear();
        }
        for (auto& d : devices) {
            if (!silent) {
                myprintf("Device ID:     %d\n", id);
                myprintf("Device name:   %s\n",
                         trim(d.getInfo<CL_DEVICE_NAME>()).c_str());
                myprintf("Device type:   %s\n",
                         opencl_dev_type_to_string(d.getInfo<CL_DEVICE_TYPE>()).c_str());
                myprintf("Device vendor: %s\n",
                          d.getInfo<CL_DEVICE_VENDOR>().c_str());
                myprintf("Device driver: %s\n",
                          d.getInfo<CL_DRIVER_VERSION>().c_str());
                myprintf("Device speed:  %u MHz\n",
                          d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
                myprintf("Device cores:  %u CU\n",
                          d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
            }

            // assign score, try to find best device
            int this_score = 0;
            std::string this_vendor = d.getInfo<CL_DEVICE_VENDOR>();
            this_score += 1000 * boost::icontains(this_vendor, "advanced micro devices");
            this_score += 1000 * boost::icontains(this_vendor, "amd");
            this_score += 1000 * boost::icontains(this_vendor, "nvidia");
            this_score +=  500 * boost::icontains(this_vendor, "intel");
            this_score +=  100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
            this_score +=  opencl_version * 10;
            if (!silent) {
                myprintf("Device score:  %d\n", this_score);
            }

            bool preferred =
                std::find(cbegin(gpus), cend(gpus), id) != cend(gpus);

            if ((this_score > best_score) || preferred) {
                best_version = opencl_version;
                best_platform = p;
                best_device = d;
                best_vendor = this_vendor;
                if (preferred) {
                    best_score = std::numeric_limits<decltype(best_score)>::max();
                } else {
                    best_score = this_score;
                }
                found_device = true;
            }
            id++;
        }
    }

    if (!found_device) {
        throw std::runtime_error("No suitable OpenCL device found.");
    }

    myprintf("Selected platform: %s\n",
        best_platform.getInfo<CL_PLATFORM_NAME>().c_str());
    myprintf("Selected device: %s\n",
        trim(best_device.getInfo<CL_DEVICE_NAME>()).c_str());
    myprintf("with OpenCL %2.1f capability.\n", best_version);

    cl::Context context;
    try {
        context = cl::Context(best_device);
    } catch (const cl::Error &e) {
        myprintf("Error creating OpenCL context: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error creating OpenCL context.");
    }
    m_context = context;
    m_device = best_device;

    // Make program of the source code in the context
    try {
        m_program = cl::Program(m_context,
                                  sourceCode_config
                                + sourceCode_convolve1
                                + sourceCode_convolve3
                                + sourceCode_sgemm);
    } catch (const cl::Error &e) {
        myprintf("Error getting kernels: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error getting OpenCL kernels.");
    }

    m_cl_args = cl_args;

    auto t = Tuner(*this, m_context, m_device);
    auto sgemm_tuners =
        t.load_sgemm_tuners(channels, WINOGRAD_P, channels, WINOGRAD_TILE);

    // Exit immediately after tuning. Some NVIDIA drivers are buggy
    // and will fail to compile the rest of the kernels after a tuning
    // run. See #729.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }

    // Build program for these specific devices
    try {
        std::string args = cl_args;
        args += sgemm_tuners;
        m_program.build(args.c_str());
    } catch (const cl::Error&) {
        myprintf("Error building kernels: %s\n",
                 m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device).c_str());
        throw std::runtime_error("Error building OpenCL kernels.");
    }

    ensure_thread_initialized();
    process_tuners(sgemm_tuners);

    m_wavefront_size =
        opencl_thread_data.m_sgemm_kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
            best_device);
    myprintf("Wavefront/Warp size: %d\n", m_wavefront_size);

    m_max_workgroup_size = best_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    m_max_workgroup_dims = best_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    myprintf("Max workgroup size: %d\n", m_max_workgroup_size);
    myprintf("Max workgroup dimensions: ");
    for (auto d : m_max_workgroup_dims) {
        myprintf("%d ", d);
    }
    myprintf("\n");

    m_init_ok = true;
}

std::string OpenCL::get_device_name() {
    std::stringstream ss;

    ss << "OpenCL: ";
    ss << m_device.getInfo<CL_DEVICE_VENDOR>() << " ";
    ss << m_device.getInfo<CL_DEVICE_NAME>() << " @ ";
    ss << m_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

    return ss.str();
}
#endif
