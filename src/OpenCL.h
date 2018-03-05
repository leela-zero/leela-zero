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

#ifndef OPENCL_H_INCLUDED
#define OPENCL_H_INCLUDED

#include "config.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION   110
#define CL_HPP_TARGET_OPENCL_VERSION    120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "Tuner.h"

static constexpr auto WINOGRAD_P = (BOARD_SIZE + 1) * (BOARD_SIZE + 1) / 4;
static constexpr auto WINOGRAD_TILE = 4 * 4;

class OpenCL;

class Layer {
    friend class OpenCL_Network;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_convolve1{false};
    std::vector<cl::Buffer> weights;
};

class ThreadData {
    friend class OpenCL;
    friend class OpenCL_Network;
private:
    bool m_is_initialized{false};
    cl::CommandQueue m_commandqueue;
    cl::Kernel m_convolve1_kernel;
    cl::Kernel m_merge_kernel;
    cl::Kernel m_in_transform_kernel;
    cl::Kernel m_sgemm_kernel;
    cl::Kernel m_out_transform_bn_kernel;
    cl::Kernel m_out_transform_bn_in_kernel;
    cl::Buffer m_inBuffer;
    cl::Buffer m_inBuffer2;
    cl::Buffer m_VBuffer;
    cl::Buffer m_MBuffer;
    cl::Buffer m_pinnedOutBuffer_pol;
    cl::Buffer m_pinnedOutBuffer_val;
    bool m_buffers_allocated{false};
};

class OpenCL_Network {
public:
    OpenCL_Network(OpenCL & opencl) : m_opencl(opencl) {}
    OpenCL & getOpenCL() {
        return m_opencl;
    }

    void push_input_convolution(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights,
                       const std::vector<float>& means,
                       const std::vector<float>& variances) {
        size_t layer = get_layer_count();
        push_weights(layer, weights);
        push_weights(layer, means);
        push_weights(layer, variances);
        m_layers[layer].is_input_convolution = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = channels;
    }

    void push_residual(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& means_1,
                       const std::vector<float>& variances_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& means_2,
                       const std::vector<float>& variances_2) {
        size_t layer = get_layer_count();
        push_weights(layer, weights_1);
        push_weights(layer, means_1);
        push_weights(layer, variances_1);
        push_weights(layer, weights_2);
        push_weights(layer, means_2);
        push_weights(layer, variances_2);
        m_layers[layer].is_residual_block = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = channels;
    }

    void push_convolve1(unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights) {
        size_t layer = get_layer_count();
        push_weights(layer, weights);
        m_layers[layer].is_convolve1 = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].channels = channels;
    }

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(const std::vector<net_t>& input,
            std::vector<net_t>& output_pol,
            std::vector<net_t>& output_val);

private:
    using weight_slice_t = std::vector<cl::Buffer>::const_iterator;

    void push_weights(size_t layer, const std::vector<float>& weights) {
        add_weights(layer, weights.size(), weights.data());
    }
    void add_weights(size_t layer, size_t size, const float* weights);

    void convolve3(int channels, int outputs,
                    cl::Buffer& bufferIn,
                    cl::Buffer& bufferOut,
                    cl::Buffer& bufferV,
                    cl::Buffer& bufferM, weight_slice_t weights,
                    cl::Buffer* bufferResidual,
                    weight_slice_t bn_weights,
                    bool skip_in_transform,
                    bool fuse_in_transform, bool store_inout);

    void convolve1(int channels, int outputs,
                  cl::Buffer& bufferInput,
                  cl::Buffer& bufferOutput,
                  cl::Buffer& bufferMerge,
                  weight_slice_t weights);

    OpenCL & m_opencl;

    // this mutex is not required for correctness, but this exists simply
    // because queue.finish() is a busy wait and having a lot of threads
    // waiting here is counterproductive CPU-wise.  At least std::mutex
    // isn't busy wait so it should be better.
    std::mutex m_queue_finish_mutex;
    std::vector<Layer> m_layers;
};

class OpenCL {
    friend class OpenCL_Network;
    friend class Tuner;
public:
    void initialize(const int channels, const std::vector<int> & gpus,
                    bool silent = false);
    void ensure_thread_initialized(void);
    std::string get_device_name();

    std::vector<size_t> get_sgemm_tuners(void);

    cl::Device m_device;
    cl::Context m_context;
private:
    void tune_sgemm(void);
    void process_tuners(std::string tuners);

    cl::Program m_program;
    std::string m_cl_args;

    struct sgemm_tuners {
        size_t mwg, nwg, kwg;
        size_t vwm, vwn;
        size_t mdimc, ndimc;
    };
    sgemm_tuners m_sgemm_tuners;
    size_t m_wavefront_size{0};
    size_t m_max_workgroup_size{0};
    std::vector<size_t> m_max_workgroup_dims;
    bool m_init_ok{false};
};

extern thread_local ThreadData opencl_thread_data;
extern const std::string sourceCode_sgemm;

#endif
