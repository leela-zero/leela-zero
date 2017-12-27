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
#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

class Layer {
    friend class OpenCL_Network;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_batchnorm{false};
    bool is_innerproduct{false};
    bool is_residual_block{false};
    std::vector<cl::Buffer> weights;
};

class ThreadData {
    friend class OpenCL;
    friend class OpenCL_Network;
private:
    bool m_is_initialized{false};
    cl::CommandQueue m_commandqueue;
    cl::Kernel m_convolve1_kernel;
    cl::Kernel m_convolve3_kernel;
    cl::Kernel m_merge_kernel;
    cl::Kernel m_batchnorm_kernel;
    cl::Buffer m_inBuffer;
    cl::Buffer m_tmpBuffer;
    cl::Buffer m_mergeBuffer;
    cl::Buffer m_outBuffer;
    cl::Buffer m_residualBuffer;
    bool m_buffers_allocated{false};
};

class OpenCL_Network {
public:
    void push_batchnorm(unsigned int spatial_size,
                        const std::vector<float>& means,
                        const std::vector<float>& variances) {
        size_t layer = get_layer_count();
        push_weights(layer, means);
        push_weights(layer, variances);
        m_layers[layer].is_batchnorm = true;
        m_layers[layer].channels = means.size();
        m_layers[layer].outputs = means.size();
        m_layers[layer].filter_size = spatial_size;
    }

    void push_convolve(unsigned int filter_size,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases) {
        size_t layer = get_layer_count();
        push_weights(layer, weights);
        push_weights(layer, biases);
        m_layers[layer].outputs = biases.size();
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = weights.size()
            / (biases.size() * filter_size * filter_size);
    }

    void push_residual(unsigned int filter_size,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& biases_1,
                       const std::vector<float>& means_1,
                       const std::vector<float>& variances_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& biases_2,
                       const std::vector<float>& means_2,
                       const std::vector<float>& variances_2) {
        size_t layer = get_layer_count();
        push_weights(layer, weights_1);
        push_weights(layer, biases_1);
        push_weights(layer, means_1);
        push_weights(layer, variances_1);
        push_weights(layer, weights_2);
        push_weights(layer, biases_2);
        push_weights(layer, means_2);
        push_weights(layer, variances_2);
        m_layers[layer].is_residual_block = true;
        m_layers[layer].outputs = biases_1.size();
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].channels = weights_1.size()
            / (biases_1.size() * filter_size * filter_size);
    }

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(const std::vector<net_t>& input, std::vector<net_t>& output);

private:
    using weight_slice_t = std::vector<cl::Buffer>::const_iterator;

    void push_weights(size_t layer, const std::vector<float>& weights) {
        add_weights(layer, weights.size(), weights.data());
    }
    void add_weights(size_t layer, size_t size, const float* weights);
    void convolve(int filter_size, int channels, int outputs,
                  cl::Buffer& input, cl::Buffer& output, cl::Buffer& merge,
                  weight_slice_t weights);
    void batchnorm(int outputs, int channel_size, cl::Buffer& input,
                   cl::Buffer& output, cl::Buffer* residual,
                   weight_slice_t weights);
    std::vector<Layer> m_layers;
};

class OpenCL {
    friend class OpenCL_Network;
public:
    void initialize();
    void ensure_thread_initialized(void);
    std::string get_device_name();

private:
    cl::Program m_program;

    size_t m_wavefront_size{0};
    size_t m_max_workgroup_size{0};
    std::vector<size_t> m_max_workgroup_dims;
    bool m_init_ok{false};
};

extern OpenCL opencl;
extern OpenCL_Network opencl_net;
extern thread_local ThreadData opencl_thread_data;

#endif
