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
#include <cassert>

#include "Tuner.h"

template <typename net_t> class OpenCL;
template <typename net_t> class OpenCL_Network;

class Layer {
    template <typename> friend class OpenCL_Network;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_convolve1{false};
    bool is_se_unit{false};
    std::vector<cl::Buffer> weights;
};

class OpenCLContext {
    template <typename> friend class OpenCL;
    template <typename> friend class OpenCL_Network;
private:
    bool m_is_initialized{false};
    cl::CommandQueue m_commandqueue;
    cl::Kernel m_convolve1_kernel;
    cl::Kernel m_merge_kernel;
    cl::Kernel m_in_transform_kernel;
    cl::Kernel m_sgemm_kernel;
    cl::Kernel m_sgemv_kernel;
    cl::Kernel m_out_transform_bn_kernel;
    cl::Kernel m_out_transform_bn_in_kernel;
    cl::Kernel m_global_avg_pooling_kernel;
    cl::Kernel m_apply_se_kernel;
    cl::Buffer m_inBuffer;
    cl::Buffer m_inBuffer2;
    cl::Buffer m_VBuffer;
    cl::Buffer m_MBuffer;
    cl::Buffer m_pool_buffer;
    cl::Buffer m_pinnedOutBuffer_pol;
    cl::Buffer m_pinnedOutBuffer_val;
    bool m_buffers_allocated{false};
};

template <typename net_t>
class OpenCL_Network {
public:
    OpenCL_Network(OpenCL<net_t> & opencl) : m_opencl(opencl) {}
    OpenCL<net_t> & getOpenCL() {
        return m_opencl;
    }

    void push_input_convolution(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<net_t>& weights,
                       const std::vector<net_t>& means,
                       const std::vector<net_t>& variances) {
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
                       const std::vector<net_t>& weights_1,
                       const std::vector<net_t>& means_1,
                       const std::vector<net_t>& variances_1,
                       const std::vector<net_t>& weights_2,
                       const std::vector<net_t>& means_2,
                       const std::vector<net_t>& variances_2) {
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

    void push_se(unsigned int channels,
                 unsigned int outputs,
                 const std::vector<net_t>& se_fc1_w,
                 const std::vector<net_t>& se_fc1_b,
                 const std::vector<net_t>& se_fc2_w,
                 const std::vector<net_t>& se_fc2_b) {
        size_t layer = get_layer_count();
        push_weights(layer, se_fc1_w);
        push_weights(layer, se_fc1_b);
        push_weights(layer, se_fc2_w);
        push_weights(layer, se_fc2_b);
        m_layers[layer].is_se_unit = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].channels = channels;
    }

    void push_convolve(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<net_t>& weights) {
        (void)filter_size;
        assert(filter_size == 1);

        size_t layer = get_layer_count();
        push_weights(layer, weights);
        m_layers[layer].is_convolve1 = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].channels = channels;
    }

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(const std::vector<float>& input,
            std::vector<float>& output_pol,
            std::vector<float>& output_val,
            OpenCLContext & opencl_context,
            const int batch_size = 1);

private:
    using weight_slice_t = std::vector<cl::Buffer>::const_iterator;

    void push_weights(size_t layer, const std::vector<net_t>& weights) {
        add_weights(layer, weights.size(), weights.data());
    }
    void add_weights(size_t layer, size_t size, const net_t* weights);

    void convolve3(OpenCLContext & opencl_context,
                    int channels, int outputs,
                    cl::Buffer& bufferIn,
                    cl::Buffer& bufferOut,
                    cl::Buffer& bufferV,
                    cl::Buffer& bufferM, weight_slice_t weights,
                    cl::Buffer* bufferResidual,
                    weight_slice_t bn_weights,
                    bool skip_in_transform,
                    bool fuse_in_transform, bool store_inout,
                    bool relu,
                    int batch_size);

    void squeeze_excitation(OpenCLContext & opencl_context,
                    int channels,
                    int fc_outputs,
                    cl::Buffer& bufferIn,
                    cl::Buffer& bufferTemp1,
                    cl::Buffer& bufferTemp2,
                    weight_slice_t weights,
                    cl::Buffer& bufferResidual,
                    int batch_size);

    void innerproduct(OpenCLContext & opencl_context,
                    const cl::Buffer& input,
                    const cl::Buffer& weights,
                    const cl::Buffer& biases,
                    cl::Buffer& output,
                    int inputs, int outputs,
                    bool relu,
                    int batch_size);

    void convolve1(OpenCLContext & opencl_context,
                  int channels, int outputs,
                  cl::Buffer& bufferInput,
                  cl::Buffer& bufferOutput,
                  cl::Buffer& bufferMerge,
                  weight_slice_t weights,
                  int batch_size);

    OpenCL<net_t> & m_opencl;

    // this mutex is not required for correctness, but this exists simply
    // because queue.finish() is a busy wait and having a lot of threads
    // waiting here is counterproductive CPU-wise.  At least std::mutex
    // isn't busy wait so it should be better.
    std::mutex m_queue_finish_mutex;
    std::vector<Layer> m_layers;
};

template <typename net_t>
class OpenCL {
    friend class OpenCL_Network<net_t>;
    friend class Tuner<net_t>;
public:
    OpenCL(int gpu, bool silent = false);

    void initialize(const int channels, size_t batch_size = 1);
    void ensure_context_initialized(OpenCLContext & opencl_context);
    std::string get_device_name();
    bool has_fp16_compute();
    bool has_tensor_cores();

    std::vector<size_t> get_sgemm_tuners();

    cl::Device m_device;
    cl::Context m_context;
private:
    void process_tuners(std::string tuners);

    size_t m_batch_size = 1;
    cl::Program m_program;
    std::string m_cl_args;

    struct sgemm_tuners {
        size_t mwg, nwg, kwg;
        size_t vwm, vwn;
        size_t mdima, ndimb;
        size_t mdimc, ndimc;
        size_t tce;
    };
    sgemm_tuners m_sgemm_tuners;
    size_t m_wavefront_size{0};
    size_t m_max_workgroup_size{0};
    std::vector<size_t> m_max_workgroup_dims;
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    bool m_init_ok{false};
};

extern const std::string sourceCode_sgemm;
extern const std::string sourceCode_common;

#endif
