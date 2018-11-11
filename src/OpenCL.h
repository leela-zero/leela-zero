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

#ifndef OPENCL_H_INCLUDED
#define OPENCL_H_INCLUDED

#include "config.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <cassert>

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
    std::vector<void*> weights;
};

class OpenCLContext {
    template <typename> friend class OpenCL;
    template <typename> friend class OpenCL_Network;
private:
    bool m_is_initialized{false};
    CUstream m_commandqueue;
    cublasHandle_t m_cublas;
    void* m_inBuffer;
    void* m_inBuffer2;
    void* m_VBuffer;
    void* m_MBuffer;
    void* m_pinnedOutBuffer_pol;
    void* m_pinnedOutBuffer_val;
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
            OpenCLContext& opencl_context,
            const int batch_size = 1);

private:
    using weight_slice_t = std::vector<void*>::const_iterator;

    void push_weights(size_t layer, const std::vector<net_t>& weights) {
        add_weights(layer, weights.size(), weights.data());
    }
    void add_weights(size_t layer, size_t size, const net_t* weights);

    void convolve3(OpenCLContext & opencl_context,
                    int channels, int outputs,
                    void* bufferIn,
                    void* bufferOut,
                    void* bufferV,
                    void* bufferM, weight_slice_t weights,
                    void** bufferResidual,
                    weight_slice_t bn_weights,
                    bool skip_in_transform,
                    bool fuse_in_transform, bool store_inout,
                    int batch_size);

    void convolve1(OpenCLContext & opencl_context,
                  int channels, int outputs,
                  void* bufferInput,
                  void* bufferOutput,
                  void* bufferMerge,
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
public:
    OpenCL(int gpu, bool silent = false);
    void initialize(const int channels);
    void ensure_context_initialized(OpenCLContext & opencl_context);
    std::string get_device_name();
    bool has_fp16_compute();

    std::vector<size_t> get_sgemm_tuners();

    CUdevice m_device;
    OpenCLContext m_context;
private:
    void process_tuners(std::string tuners);
    struct sgemm_tuners {
        size_t mwg, nwg, kwg;
        size_t vwm, vwn;
        size_t mdimc, ndimc;
    };
    sgemm_tuners m_sgemm_tuners;
    size_t m_wavefront_size{0};
    size_t m_max_workgroup_size{0};
    std::vector<size_t> m_max_workgroup_dims;
    bool m_fp16_compute{false};
    bool m_init_ok{false};
};

#endif
