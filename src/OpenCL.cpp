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

#include <assert.h>
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

using namespace Utils;

static std::string sourceCode_config = R"(
    #ifdef USE_HALF
    typedef half net_t;
    #define vload_net_t(offset,p) vload_half(offset,p)
    #define vstore_net_t(data,offset,p) vstore_half(data,offset,p)
    #else
    typedef float net_t;
    #define vload_net_t(offset,p) ((p)[(offset)])
    #define vstore_net_t(data,offset,p) (((p)[(offset)])=(data))
    #endif
)";

static std::string sourceCode_convolve3 = R"(
__kernel void in_transform(__global net_t *in, __global float *V, const int C) {

    const int W = 19;
    const int H = 19;
    const int WTILES = (W + 1) / 2;
    const int P = WTILES*WTILES;

    const int block = get_global_id(0);
    const int ch = get_global_id(1);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    //Tiles overlap by 2
    const int yin = 2 * block_y - 1;
    const int xin = 2 * block_x - 1;

    if (block < P && ch < C) {

        //Cache input tile and handle zero padding
        float x[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if ((yin+i) >= 0 && (xin+j) >= 0 && (yin+i) < H && (xin+j) < W) {
                    x[i][j] = vload_net_t(ch*(W*H) + (yin+i)*W + (xin+j), in);
                } else {
                    x[i][j] = 0.0f;
                }
            }
        }

        float T1[4][4];
        float T2[4][4];

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

        T2[0][0] = T1[0][0] - T1[0][2];
        T2[0][1] = T1[0][1] + T1[0][2];
        T2[0][2] = T1[0][2] - T1[0][1];
        T2[0][3] = T1[0][1] - T1[0][3];
        T2[1][0] = T1[1][0] - T1[1][2];
        T2[1][1] = T1[1][1] + T1[1][2];
        T2[1][2] = T1[1][2] - T1[1][1];
        T2[1][3] = T1[1][1] - T1[1][3];
        T2[2][0] = T1[2][0] - T1[2][2];
        T2[2][1] = T1[2][1] + T1[2][2];
        T2[2][2] = T1[2][2] - T1[2][1];
        T2[2][3] = T1[2][1] - T1[2][3];
        T2[3][0] = T1[3][0] - T1[3][2];
        T2[3][1] = T1[3][1] + T1[3][2];
        T2[3][2] = T1[3][2] - T1[3][1];
        T2[3][3] = T1[3][1] - T1[3][3];

        const int offset = ch*P + block;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                V[(i*4 + j)*C*P + offset] = T2[i][j];
            }
        }
    }
}

__kernel void out_transform(__global float *M, __global net_t *Y, int K) {

    const int W = 19;
    const int H = 19;
    const int WTILES = (W + 1) / 2;
    const int P = WTILES * WTILES;

    int block = get_global_id(0);
    int k = get_global_id(1);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    int x = 2*block_x;
    int y = 2*block_y;

    if (k < K && block < P) {
        int b = block_y * WTILES + block_x;
        float temp_m[16];
        for (int xi = 0; xi < 4; xi++) {
            for (int nu = 0; nu < 4; nu++) {
                temp_m[xi*4 + nu] = M[xi*(4*K*P) + nu*(K*P)+ k*P + b];
            }
        }

        float o11 = temp_m[0*4 + 0] + temp_m[0*4 + 1] + temp_m[0*4 + 2] +
                    temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] +
                    temp_m[2*4 + 0] + temp_m[2*4 + 1] + temp_m[2*4 + 2];

        float o12 = temp_m[0*4 + 1] - temp_m[0*4 + 2] - temp_m[0*4 + 3] +
                    temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] +
                    temp_m[2*4 + 1] - temp_m[2*4 + 2] - temp_m[2*4 + 3];

        float o21 = temp_m[1*4 + 0] + temp_m[1*4 + 1] + temp_m[1*4 + 2] -
                    temp_m[2*4 + 0] - temp_m[2*4 + 1] - temp_m[2*4 + 2] -
                    temp_m[3*4 + 0] - temp_m[3*4 + 1] - temp_m[3*4 + 2];

        float o22 = temp_m[1*4 + 1] - temp_m[1*4 + 2] - temp_m[1*4 + 3] -
                    temp_m[2*4 + 1] + temp_m[2*4 + 2] + temp_m[2*4 + 3] -
                    temp_m[3*4 + 1] + temp_m[3*4 + 2] + temp_m[3*4 + 3];

        vstore_net_t(o11, k*(H*W) + (y)*W + (x), Y);
        if (x+1 < W) {
            vstore_net_t(o12, k*(H*W) + (y)*W + (x+1), Y);
        }
        if (y+1 < H) {
            vstore_net_t(o21, k*(H*W) + (y+1)*W + (x), Y);
            if (x+1 < W) {
                vstore_net_t(o22, k*(H*W) + (y+1)*W + (x+1), Y);
            }
        }
    }
}

__kernel void out_transform_fused_bn(__global float *M,
                __global net_t *Y,
                int K,
                __global const net_t * residual,
                __constant const net_t * means,
                __constant const net_t * stddivs) {

    const int W = 19;
    const int H = 19;
    const int WTILES = (W + 1) / 2;
    const int P = WTILES * WTILES;

    int block = get_global_id(0);
    int k = get_global_id(1);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    int x = 2*block_x;
    int y = 2*block_y;

    if (k < K && block < P) {
        int b = block_y * WTILES + block_x;
        float temp_m[16];
        for (int xi = 0; xi < 4; xi++) {
            for (int nu = 0; nu < 4; nu++) {
                temp_m[xi*4 + nu] = M[xi*(4*K*P) + nu*(K*P)+ k*P + b];
            }
        }

        float o[4];
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

        const float mean = vload_net_t(k, means);
        const float scale_stddiv = vload_net_t(k, stddivs);

        for (int i = 0; i < 4; i++) {
            o[i] = scale_stddiv * (o[i] - mean);
        }

        const bool pred[4] = { 1, x+1 < W, y+1 < H, x+1 < W & y+1 < H};

        const int a[4] = {(y)*W + (x), (y)*W + (x+1), (y+1)*W + (x), (y+1)*W + (x+1)};

        for (int i = 0; i < 4; i++) {
            if (pred[i]) {
                if (residual) {
                    o[i] += vload_net_t(k*(H*W) + a[i], residual);
                }
                o[i] = o[i] > 0 ? o[i] : 0.0f;
                vstore_net_t(o[i], k*(H*W) + a[i], Y);
            }
        }
    }
}
)";

static std::string sourceCode_utility = R"(
    __kernel void batchnorm(
                        __global const net_t * in,
                        __global net_t * out,
                        __global const net_t * residual,
                        __constant const net_t * means,
                        __constant const net_t * stddivs) {

        // cl::NDRange global(outputs, 19*19);
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);

        const int output = gx;
        const int outputs      = get_global_size(0);
        const int channel_size = get_global_size(1);

        const unsigned int o = output;
        const unsigned int b = gy;

        const float mean = vload_net_t(o, means);
        const float scale_stddiv = vload_net_t(o, stddivs);

        // BN
        float sum = scale_stddiv * (vload_net_t(o * channel_size + b, in) - mean);
        // Residual Eltwise
        if (residual) {
            sum += vload_net_t(o * channel_size + b, residual);
        }
        // ReLU
        vstore_net_t(sum > 0 ? sum : 0.0f, o * channel_size + b, out);
    }
)";


size_t lcm(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }
    size_t ret = a + (b - a % b);
    assert(ret % b == 0);
    return ret;
}

OpenCL opencl;
OpenCL_Network opencl_net;
thread_local ThreadData opencl_thread_data;

void OpenCL::ensure_thread_initialized() {
    if (!opencl_thread_data.m_is_initialized) {
        // Make kernels
        opencl_thread_data.m_in_transform_kernel = cl::Kernel(m_program, "in_transform");
        opencl_thread_data.m_sgemm_kernel = cl::Kernel(m_program, "XgemmDirectBatchedTN");
        opencl_thread_data.m_out_transform_kernel = cl::Kernel(m_program, "out_transform");
        opencl_thread_data.m_out_transform_bn_kernel = cl::Kernel(m_program, "out_transform_fused_bn");
        opencl_thread_data.m_batchnorm_kernel = cl::Kernel(m_program, "batchnorm");
        opencl_thread_data.m_commandqueue = cl::CommandQueue(cl::Context::getDefault(),
                                                             cl::Device::getDefault());
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
        CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
        weightSize,
        const_cast<net_t*>(converted_weights.data()));
}

void OpenCL_Network::forward(const std::vector<net_t>& input,
                             std::vector<net_t>& output) {
    constexpr auto width = 19;
    constexpr auto height = 19;
    constexpr auto tiles = (width + 1)*(height + 1) / 4;
    constexpr auto one_plane = width * height * sizeof(net_t);
    constexpr auto one_filter = 4 * 4 * sizeof(net_t);

    opencl.ensure_thread_initialized();

    if (!opencl_thread_data.m_buffers_allocated) {
        unsigned int max_channels = 0;
        for (const auto& layer : m_layers) {
            max_channels = std::max(max_channels,
                    std::max(layer.channels, layer.outputs));
        }
        const auto alloc_inSize = one_plane *  max_channels;
        const auto alloc_vm_size = tiles * max_channels * one_filter;

        opencl_thread_data.m_inBuffer = cl::Buffer(
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_tmpBuffer = cl::Buffer(
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_residualBuffer = cl::Buffer(
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_VBuffer = cl::Buffer(
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_vm_size);
        opencl_thread_data.m_MBuffer = cl::Buffer(
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_vm_size);
        opencl_thread_data.m_buffers_allocated = true;
    }

    cl::Buffer & inBuffer = opencl_thread_data.m_inBuffer;
    cl::Buffer & tmpBuffer = opencl_thread_data.m_tmpBuffer;
    cl::Buffer & VBuffer = opencl_thread_data.m_VBuffer;
    cl::Buffer & MBuffer = opencl_thread_data.m_MBuffer;
    cl::Buffer & residualBuffer = opencl_thread_data.m_residualBuffer;
    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    const auto inSize = sizeof(net_t) * input.size();
    queue.enqueueWriteBuffer(inBuffer, CL_FALSE, 0, inSize, input.data());

    for (const auto& layer : m_layers) {
        if (layer.is_batchnorm) {
            auto bn_weights = begin(layer.weights);
            batchnorm(layer.outputs,
                      layer.filter_size,
                      inBuffer,
                      tmpBuffer,
                      nullptr,
                      bn_weights);
            std::swap(inBuffer, tmpBuffer);
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            auto conv1_weights = begin(layer.weights);
            auto bn1_weights   = begin(layer.weights) + 2;
            auto conv2_weights = begin(layer.weights) + 4;
            auto bn2_weights   = begin(layer.weights) + 6;
            const auto inBufferSize = layer.channels * one_plane;
            queue.enqueueCopyBuffer(inBuffer, residualBuffer, 0, 0, inBufferSize);
            convolve3(layer.channels,
                     layer.outputs,
                     inBuffer,
                     VBuffer,
                     MBuffer,
                     conv1_weights,
                     nullptr,
                     &bn1_weights);
            convolve3(layer.channels,
                     layer.outputs,
                     inBuffer,
                     VBuffer,
                     MBuffer,
                     conv2_weights,
                     &residualBuffer,
                     &bn2_weights);
        } else  {
            auto conv_weights = begin(layer.weights);
            // plain convolution
            convolve3(layer.channels,
                     layer.outputs,
                     inBuffer,
                     VBuffer,
                     MBuffer,
                     conv_weights,
                     nullptr,
                     nullptr);
        }
    }

    const auto finalSize = m_layers.back().outputs * one_plane;
    queue.enqueueReadBuffer(inBuffer, CL_FALSE, 0, finalSize, output.data());

    queue.finish();
}

void OpenCL_Network::convolve3(int channels, int outputs,
                              cl::Buffer& bufferInOut,
                              cl::Buffer& bufferV,
                              cl::Buffer& bufferM,
                              weight_slice_t weights,
                              cl::Buffer* bufferResidual,
                              weight_slice_t* bn_weights) {

    cl::Kernel in_transform_kernel = opencl_thread_data.m_in_transform_kernel;
    cl::Kernel sgemm_kernel = opencl_thread_data.m_sgemm_kernel;
    cl::Kernel out_transform_kernel = opencl_thread_data.m_out_transform_kernel;
    cl::Kernel out_transform_bn_kernel = opencl_thread_data.m_out_transform_bn_kernel;

    auto wgd = opencl.m_sgemm_wgd;
    auto mdimcd = opencl.m_sgemm_mdimcd;
    auto ndimcd = opencl.m_sgemm_ndimcd;
    auto wavefront_size = opencl.m_wavefront_size;

    assert(wgd != 0);
    assert(ndimcd != 0);
    assert(mdimcd != 0);
    assert(wavefront_size != 0);

    constexpr int tiles = (19 + 1) * (19 + 1) / 4;

    auto wgs = lcm(tiles, wavefront_size);

    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    try {
        in_transform_kernel.setArg(0, bufferInOut);
        in_transform_kernel.setArg(1, bufferV);
        in_transform_kernel.setArg(2, channels);

        queue.enqueueNDRangeKernel(in_transform_kernel, cl::NullRange,
                                   cl::NDRange(wgs, channels));
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3: " << e.what() << ": "
	        << e.err() << std::endl;
        throw;
    }

    try {
        sgemm_kernel.setArg(0, outputs);
        sgemm_kernel.setArg(1, tiles);
        sgemm_kernel.setArg(2, channels);
        sgemm_kernel.setArg(3, weights[0]);
        sgemm_kernel.setArg(4, bufferV);
        sgemm_kernel.setArg(5, bufferM);

        cl::NDRange local_sgemm = {mdimcd, ndimcd, 1};

        auto m_ceil = lcm(outputs, wgd);
        auto n_ceil = lcm(tiles, wgd);

        cl::NDRange size_sgemm = {(m_ceil * mdimcd) / wgd,
                                  (n_ceil * ndimcd) / wgd,
                                  16};

        queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange,
                                    size_sgemm, local_sgemm);
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3: " << e.what() << ": "
	        << e.err() << std::endl;
        throw;
    }

    try {
        if (bn_weights) {
            out_transform_bn_kernel.setArg(0, bufferM);
            out_transform_bn_kernel.setArg(1, bufferInOut);
            out_transform_bn_kernel.setArg(2, outputs);
            if (bufferResidual) {
                out_transform_bn_kernel.setArg(3, *bufferResidual);
            } else {
                out_transform_bn_kernel.setArg(3, nullptr);
            }
            out_transform_bn_kernel.setArg(4, (*bn_weights)[0]);
            out_transform_bn_kernel.setArg(5, (*bn_weights)[1]);

            queue.enqueueNDRangeKernel(out_transform_bn_kernel, cl::NullRange,
                                       cl::NDRange(wgs, outputs));
        } else {
            out_transform_kernel.setArg(0, bufferM);
            out_transform_kernel.setArg(1, bufferInOut);
            out_transform_kernel.setArg(2, outputs);

            queue.enqueueNDRangeKernel(out_transform_kernel, cl::NullRange,
                                       cl::NDRange(wgs, outputs));
        }
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3: " << e.what() << ": "
	        << e.err() << std::endl;
        throw;
    }

}

void OpenCL_Network::batchnorm(int outputs,
                               int channel_size,
                               cl::Buffer& bufferInput,
                               cl::Buffer& bufferOutput,
                               cl::Buffer* bufferResidual,
                               weight_slice_t weights) {
    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    cl::Kernel & batchnorm_kernel = opencl_thread_data.m_batchnorm_kernel;

    size_t channelGroup = 1;
    if (channel_size == 361) {
        channelGroup = 19;
    }

    try {
        batchnorm_kernel.setArg(0, bufferInput);
        batchnorm_kernel.setArg(1, bufferOutput);
        if (bufferResidual) {
            batchnorm_kernel.setArg(2, *bufferResidual);
        } else {
            batchnorm_kernel.setArg(2, nullptr);
        }
        batchnorm_kernel.setArg(3, weights[0]);
        batchnorm_kernel.setArg(4, weights[1]);

        queue.enqueueNDRangeKernel(batchnorm_kernel, cl::NullRange,
                                   cl::NDRange(outputs, channel_size),
                                   cl::NDRange(std::min(8, outputs), channelGroup));
    } catch (const cl::Error &e) {
        std::cerr << "Error in batchnorm: " << e.what() << ": "
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

    bool wgd = false;
    bool mdimcd = false;
    bool ndimcd = false;
    while (ss >> buf) {
        found = buf.find("=");
        if (found == std::string::npos) {
            std::cerr << "Invalid tuner string: " << tuners << std::endl;
            std::exit(-1);
        }
        std::string name = buf.substr(0, found);
        auto value = std::stoi(buf.substr(found+1, std::string::npos));
        if (name == "WGD") {
            m_sgemm_wgd = value;
            wgd = true;
        }
        if (name == "MDIMCD") {
            m_sgemm_mdimcd = value;
            mdimcd = true;
        }
        if (name == "NDIMCD") {
            m_sgemm_ndimcd = value;
            ndimcd = true;
        }
    }
    if (!wgd || !mdimcd || !ndimcd) {
        std::cerr << "Missing tuner parameters";
        if (!wgd) {
            std::cerr << " WGD";
        }
        if (!mdimcd) {
            std::cerr << " MDIMCD";
        }
        if (!ndimcd) {
            std::cerr << " NDIMCD";
        }
        std::cerr << std::endl;
        std::exit(-1);
    }
}

void OpenCL::initialize(void) {
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error &e) {
        myprintf("OpenCL: %s\n", e.what());
        throw;
    }

    float best_version = 0.0f;
    cl::Platform best_platform;
    cl::Device best_device;
    std::string best_vendor;
    int best_score = 0;
    bool found_device = false;
    int id = 0;

    myprintf("Detected %d OpenCL platforms\n", platforms.size());

    for (const auto &p : platforms) {
        std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
        std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
        std::string platname = p.getInfo<CL_PLATFORM_NAME>();
        std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
        myprintf("Platform version: %s\n", platvers.c_str());;
        myprintf("Platform profile: %s\n", platprof.c_str());
        myprintf("Platform name:    %s\n", platname.c_str());
        myprintf("Platform vendor:  %s\n", platvend.c_str());

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

            // assign score, try to find best device
            int this_score = 0;
            std::string this_vendor = d.getInfo<CL_DEVICE_VENDOR>();
            this_score += 1000 * boost::icontains(this_vendor, "advanced micro devices");
            this_score += 1000 * boost::icontains(this_vendor, "amd");
            this_score += 1000 * boost::icontains(this_vendor, "nvidia");
            this_score +=  500 * boost::icontains(this_vendor, "intel");
            this_score +=  100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
            this_score +=  opencl_version * 10;
            myprintf("Device score:  %d\n", this_score);

            bool preferred = std::find(cfg_gpus.cbegin(), cfg_gpus.cend(), id) != cfg_gpus.cend();

            if ((this_score > best_score) || preferred) {
                best_version = opencl_version;
                best_platform = p;
                best_device = d;
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

    cl::Platform::setDefault(best_platform);
    myprintf("Selected platform: %s\n", best_platform.getInfo<CL_PLATFORM_NAME>().c_str());
    myprintf("Selected device: %s\n", trim(best_device.getInfo<CL_DEVICE_NAME>()).c_str());
    myprintf("with OpenCL %2.1f capability\n", best_version);

    cl::Context context;
    try {
        context = cl::Context(best_device);
    } catch (const cl::Error &e) {
        myprintf("Error creating OpenCL context: %s: %d", e.what(), e.err());
        throw;
    }
    cl::Context::setDefault(context);
    cl::Device::setDefault(best_device);

    // Read source file
    //std::ifstream sourceFile("convolve_kernel.cl", std::ifstream::in);
    //std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),
    //                       (std::istreambuf_iterator<char>()));

    // Make program of the source code in the context
	std::string sourceCode_sgemm =
        #include "clblast_level3/common.opencl"
        #include "clblast_level3/level3.opencl"
        #include "clblast_level3/xgemm_direct_part1.opencl"
        #include "clblast_level3/xgemm_direct_part2.opencl"
        #include "clblast_level3/xgemm_direct_part3.opencl"
        #include "clblast_level3/xgemm_direct_batched.opencl"
	;

    try {
        m_program = cl::Program(sourceCode_config
                                + sourceCode_convolve3
                                + sourceCode_utility
                                + sourceCode_sgemm);
    } catch (const cl::Error &e) {
        myprintf("Error getting kernels: %s: %d", e.what(), e.err());
        throw;
    }
    // Build program for these specific devices
    try {
	    std::string args = "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";

        std::string buf;
        std::stringstream ss(SGEMM_TUNERS);

        while (ss >> buf) {
            args += " -D" + buf;
        }

#ifdef USE_HALF
        args += " -DUSE_HALF";
#endif
        m_program.build(args.c_str());
    } catch (const cl::Error&) {
        myprintf("Error building kernels: %s\n",
                    m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()).c_str());
        throw;
    }

    ensure_thread_initialized();

    process_tuners(SGEMM_TUNERS);

    m_wavefront_size =
        opencl_thread_data.m_batchnorm_kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
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

    cl::Device device = cl::Device::getDefault();
    ss << "OpenCL: ";
    ss << device.getInfo<CL_DEVICE_VENDOR>() << " ";
    ss << device.getInfo<CL_DEVICE_NAME>() << " @ ";
    ss << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

    return ss.str();
}
#endif
