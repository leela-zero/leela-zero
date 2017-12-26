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
#include <chrono>
#include <boost/lexical_cast.hpp>

#include "Network.h"
#include "GTP.h"
#include "Utils.h"

using namespace Utils;

static std::string sourceCode_config = R"(
    // vfloat_t : float if normal, float4 if compiled for batch-of-4
    #if (BATCH_SIZE == 8)
        typedef float8 vfloat_t;
    #elif (BATCH_SIZE == 4)
        typedef float4 vfloat_t;
    #elif (BATCH_SIZE == 2)
        typedef float2 vfloat_t;
    #else
        typedef float vfloat_t;
    #endif

    #ifdef USE_HALF
        typedef half net_t;
        #define vload_net_t(offset,p) vload_half(offset,p)
        #define vstore_net_t(data,offset,p) vstore_half(data,offset,p)
        #if (BATCH_SIZE == 8)
            typedef struct __vnet_t {
                half x; half y; half z; half w; half a; half b; half c; half d;
            } vnet_t;
            #define vload_vnet_t(offset,p) vload_half8(offset,(__global half*)(p))
            #define vstore_vnet_t(data,offset,p) vstore_half8(data,offset,(__global half*)(p))
        #elif (BATCH_SIZE == 4)
            typedef struct __vnet_t {
                half x; half y; half z; half w;
            } vnet_t;
            #define vload_vnet_t(offset,p) vload_half4(offset,(__global half*)(p))
            #define vstore_vnet_t(data,offset,p) vstore_half4(data,offset,(__global half*)(p))
        #elif (BATCH_SIZE == 2)
            typedef struct __vnet_t {
                half x; half y;
            } vnet_t;
            #define vload_vnet_t(offset,p) vload_half2(offset,(__global half*)(p))
            #define vstore_vnet_t(data,offset,p) vstore_half2(data,offset,(__global half*)(p))
        #else // BATCH_SIZE == 1
            typedef net_t vnet_t;
            #define vload_vnet_t(offset,p) vload_net_t(offset,p)
            #define vstore_vnet_t(data,offset,p) vstore_net_t(data,offset,p)
        #endif
    #else // !USE_HALF
        typedef float net_t;
        #define vload_net_t(offset,p) ((p)[(offset)])
        #define vstore_net_t(data,offset,p) (((p)[(offset)])=(data))
        #if (BATCH_SIZE == 8)
            typedef float8 vnet_t;
            #define vload_vnet_t(offset,p) ((p)[(offset)])
            #define vstore_vnet_t(data,offset,p) (((p)[(offset)])=(data))
        #elif (BATCH_SIZE == 4)
            typedef float4 vnet_t;
            #define vload_vnet_t(offset,p) ((p)[(offset)])
            #define vstore_vnet_t(data,offset,p) (((p)[(offset)])=(data))
        #elif (BATCH_SIZE == 2)
            typedef float2 vnet_t;
            #define vload_vnet_t(offset,p) ((p)[(offset)])
            #define vstore_vnet_t(data,offset,p) (((p)[(offset)])=(data))
        #else
            typedef net_t vnet_t;
            #define vload_vnet_t(offset,p) vload_net_t(offset,p)
            #define vstore_vnet_t(data,offset,p) vstore_net_t(data,offset,p)
        #endif
    #endif
)";
static std::string sourceCode_convolve1 = R"(
    __kernel
    __attribute__((work_group_size_hint(8, 16, 1)))
    void convolve1(
                   __global const vnet_t * in,
                   __global vnet_t * merge,
                   __global const net_t * weights,
                   __local vfloat_t * channel_buff,
                   __local vfloat_t * row_buff) {
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

        const int width = 19;
        const int height = 19;
        const int strip_size = width;

        // Copy the input channels (strips) locally
        if (out_buff_size < 19 && ly == 0) {
            // strip-row
            for (int w = 0; w < width; w++) {
                channel_buff[lx * width + w] =
                    vload_vnet_t((c * height + row) * width + w, in);
            }
        } else if (out_buff_size >= 19 && ly < 19) {
            // Every thread copies a column
            channel_buff[lx * width + ly] = vload_vnet_t((c * height + row) * width + ly, in);
        }

        // Copy the filter we are applying locally
        __private float filter_buff = vload_net_t((o * channels + c), weights);

        barrier(CLK_LOCAL_MEM_FENCE);

        int out_lane = 0;
        int out_cw   = 0;
        #pragma unroll
        for (int cw = 0; cw < width; cw++) {
            int fid = lx * strip_size;
            vfloat_t out  = channel_buff[fid + cw] * filter_buff;
            row_buff[(ly * chan_buff_size + lx) * row_buff_size + out_lane] = out;
            out_lane++;
            // Row buffer full or last lane?
            if (out_lane == row_buff_size || (cw == width - 1)) {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (lx < out_lane) {
                    vfloat_t val;
                    val  = row_buff[(ly * chan_buff_size + 0) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 1) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 2) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 3) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 4) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 5) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 6) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 7) * row_buff_size + lx];
                    vstore_vnet_t(val, (((c >> chan_shift) * height + row) * width + out_cw + lx) * outputs + o, merge);
                }
                out_cw  += row_buff_size;
                out_lane = 0;
           }
       }
    }
)";

static std::string sourceCode_convolve3 = R"(
    __kernel
    __attribute__((work_group_size_hint(8, 32, 1)))
    void convolve3(
                   __global const vnet_t * in,
                   __global vnet_t * merge,
                   __global const net_t * weights,
                   __local vfloat_t * channel_buff,
                   __local vfloat_t * row_buff,
                   const int row_tile_size,
                   const int row_buff_size,
                   const int chan_buff_size,
                   const int chan_shift) {

        // cl::NDRange global(channels, outputs, row);
        const int c   = get_global_id(0);  // channel
        const int o   = get_global_id(1);  // output
        const int r   = get_global_id(2);  // row

        const int channels = get_global_size(0);
        const int outputs  = get_global_size(1);

        // cl::NDRange local(2, (1->32), 1);
        const int lx = get_local_id(0);
        const int ly = get_local_id(1);

        const int out_buff_size  = get_local_size(1);
        const int width = 19;
        const int height = 19;

        const int filter_size = 3;
        const int filter_len = filter_size * filter_size;
        const int mid = (filter_size / 2) + 1;
        const int extent = mid - 1;
        const int pad_width = width + filter_size - 1;

        // input = channels * height * width
        // output = outputs * height * width
        // weights = output * channels * filter
        // merge = channels * outputs * height * width

        __private float filter_buff[9];
        __private vfloat_t chan_cache[2];
        __private vfloat_t stripe_cache[9];

        // Copy the filter we are applying locally
        // output * channel * filter_len
        for (int f = 0; f < filter_len; f++) {
            filter_buff[f] = vload_net_t((o * channels + c) * filter_len + f, weights);
        }

        for (int tile = 0; tile < row_tile_size; tile++) {
            int row = r * row_tile_size + tile;
            if (row > 18) break;

            // Copy the input channels (strips) locally
            if (out_buff_size < 21 && ly == 0) {
                // strip-row
                for (int srow = 0; srow < filter_size; srow++) {
                    int in_row = row - extent + srow;
                    channel_buff[(lx * pad_width + 0) * filter_size + srow]             = 0.0f;
                    if ((unsigned)in_row < height) {
                        for (int w = 0; w < width; w++) {
                            vfloat_t val = vload_vnet_t((c * height + in_row) * width + w, in);
                            channel_buff[(lx * pad_width + w + extent) * filter_size + srow] = val;
                        }
                    } else {
                        for (int w = 0; w < width; w++) {
                            channel_buff[(lx * pad_width + w + extent) * filter_size + srow] = 0.0f;
                        }
                    }
                    channel_buff[(lx * pad_width + pad_width - 1) * filter_size + srow] = 0.0f;
                }
            } else if (out_buff_size >= 21 && ly < 21) {
                // Every thread copies a column
                int copy_idx = (lx * pad_width + ly) * filter_size;
                if (tile == 0 || row == 18) {
                    // Every thread copies a column
                    for (int srow = 0; srow < filter_size; srow++) {
                        int in_row = row - extent + srow;
                        vfloat_t val = 0.0f;
                        if ((unsigned)in_row < height && ly >= 1 && ly <= 19) {
                            val = vload_vnet_t((c * height + in_row) * width + ly - 1, in);
                        }
                        channel_buff[copy_idx + srow] = val;
                        if (srow > 0) {
                            chan_cache[srow - 1] = val;
                        }
                    }
                } else {
                    int in_row = row - extent + 2;
                    vfloat_t val = 0.0f;
                    if (ly >= 1 && ly <= 19) {
                        val = vload_vnet_t((c * height + in_row) * width + ly - 1, in);
                    }
                    channel_buff[copy_idx + 0] = chan_cache[0];
                    channel_buff[copy_idx + 1] = chan_cache[1];
                    channel_buff[copy_idx + 2] = val;
                    chan_cache[0] = chan_cache[1];
                    chan_cache[1] = val;
                }
            }

            int out_lane = 0;
            int out_cw   = 0;
            __local vfloat_t * out_row_buff = &row_buff[(ly * chan_buff_size + lx) * row_buff_size];
            int fid = (lx * pad_width) * filter_size;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int rc = 0; rc < 9; rc++) {
                stripe_cache[rc] = channel_buff[fid + rc];
            }

            #pragma unroll
            for (int cw = 0; cw < width; cw++) {
                // Start filter
                vfloat_t out  =   stripe_cache[      0] * filter_buff[0]
                             + stripe_cache[      1] * filter_buff[3]
                             + stripe_cache[      2] * filter_buff[6]
                             + stripe_cache[      3] * filter_buff[1]
                             + stripe_cache[      4] * filter_buff[4]
                             + stripe_cache[      5] * filter_buff[7]
                             + stripe_cache[      6] * filter_buff[2]
                             + stripe_cache[      7] * filter_buff[5]
                             + stripe_cache[      8] * filter_buff[8];
                // End filter
                out_row_buff[out_lane++] = out;
                fid += filter_size;

                for (int rc = 0; rc < 6; rc++) {
                    stripe_cache[rc] = stripe_cache[rc + 3];
                }
                stripe_cache[6] = channel_buff[fid + 6];
                stripe_cache[7] = channel_buff[fid + 7];
                stripe_cache[8] = channel_buff[fid + 8];

                // Row buffer full or last lane?
                if (out_lane == row_buff_size || (cw == width - 1)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (lx < out_lane) {
                        // lx = channels 2 or 8, ly = outputs 32
                        // repurpose the lx threads over columns now
                        if (chan_buff_size == 8) {
                            vfloat_t val;
                            val  = row_buff[(ly * chan_buff_size + 0) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 1) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 2) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 3) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 4) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 5) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 6) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 7) * row_buff_size + lx];
                            vstore_vnet_t(val, (((c >> chan_shift) * height + row) * width + out_cw + lx) * outputs + o, merge);
                        } else if (chan_buff_size == 2) {
                            vfloat_t val;
                            val  = row_buff[(ly * chan_buff_size + 0) * row_buff_size + lx];
                            val += row_buff[(ly * chan_buff_size + 1) * row_buff_size + lx];
                            vstore_vnet_t(val, (((c >> chan_shift) * height + row) * width + out_cw + lx) * outputs + o, merge);
                        }
                    }
                    out_cw  += row_buff_size;
                    out_lane = 0;
                }
            }
        }
    }
)";

static std::string sourceCode_utility = R"(
    __kernel void merge(
                        __global const vnet_t * in,
                        __global vnet_t * out,
                        __constant const net_t * biases,
                        __private const int channels) {

        // cl::NDRange global(outputs, 19*19);
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);

        const int output = gx;
        const int b = gy;
        const int outputs = get_global_size(0);

        const int width = 19;
        const int height = 19;
        const int boardsize = width * height;

        const int o = output;
        const float bias = vload_net_t(o, biases);

        vfloat_t sum = bias;
        for (int c = 0; c < channels; c++) {
            sum += vload_vnet_t((c * boardsize + b) * outputs + o, in);
        }
        vstore_vnet_t(sum, o * boardsize + b, out);
    }

    __kernel void batchnorm(
                        __global const vnet_t * in,
                        __global vnet_t * out,
                        __global const vnet_t * residual,
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
        vfloat_t sum = scale_stddiv * (vload_vnet_t(o * channel_size + b, in) - mean);
        // Residual Eltwise
        if (residual) {
            sum += vload_vnet_t(o * channel_size + b, residual);
        }
        // ReLU
	vfloat_t v = sum > 0 ? sum : 0.0f;
        vstore_vnet_t(v, o * channel_size + b, out);
    }
)";

OpenCL opencl;
OpenCL_Network opencl_net;
thread_local ThreadData opencl_thread_data;

void OpenCL::ensure_thread_initialized() {
    if (!opencl_thread_data.m_is_initialized) {
        // Make kernels
        for(const auto & x : m_program) {
            opencl_thread_data.m_convolve1_kernel[x.first] = cl::Kernel(x.second, "convolve1");
            opencl_thread_data.m_convolve3_kernel[x.first] = cl::Kernel(x.second, "convolve3");
            opencl_thread_data.m_merge_kernel[x.first] = cl::Kernel(x.second, "merge");
            opencl_thread_data.m_batchnorm_kernel[x.first] = cl::Kernel(x.second, "batchnorm");
        }
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
                             std::vector<float>& output) {
#ifndef USE_OPENCL_BATCHING
    // we don't need workers to group workloads on a batch size of 1.
    // directly call run_forward
    const std::vector<net_t> * inptr = &input;
    std::vector<float_t> * outptr = &output;
    run_forward(&inptr, &outptr, 1);

#else
    if(!m_workers_launched) {
        // test run each batch size.  pick ones that successfully passed sanity check
        std::list<unsigned int> valid_batches;
        for(const auto & x : opencl.m_program) {
            unsigned int batch_size = x.first;
            try {
                myprintf("OpenCL: testing batch size %d\n", batch_size);
                run_forward(nullptr, nullptr, batch_size);
                valid_batches.push_back(batch_size);
            } catch(cl::Error &) {
                myprintf("OpenCL: failed batch size %d and dropping\n", batch_size);
            }
        }

        // launch the worker thread.  2 threads so that we can fully utilize GPU, since the 
        // worker thread consists of some CPU work for task preparation.
        constexpr int num_threads = 2;
        for(int i=0; i<num_threads; i++) {
            std::thread worker( [this, valid_batches]{
                while(true) {
                    std::unique_lock<std::mutex> lk(m_task_mutex);

                    unsigned int max_batchsize = valid_batches.back();

                    // the 50ms timeout is based on a wild guess, assuming we will not run out of evaluation operations
                    // due to the sheer amount of concurrent threads.  timeouts will probably only happen
                    // on the final evaluations of the move.
                    m_task_cond.wait_for(lk, std::chrono::milliseconds(50), [this,max_batchsize]{
                        return (m_task_queue.size() >= max_batchsize); 
                    });
               
                    if(m_task_queue.empty()) {
                        lk.unlock();
                        continue;
                    }

                    unsigned int batch_size = 1;
                    for(auto & x : valid_batches) {
                        if(m_task_queue.size() < x) {
                            break;
                        }
                        batch_size = x;
                    } 

                    unsigned int count = 0;
                    ForwardTask tasks[batch_size];
                    while(count < batch_size && !m_task_queue.empty()) {
                        tasks[count++] = std::move(m_task_queue.front());
                        m_task_queue.pop_front();
                    }
                    lk.unlock();
    
                    const std::vector<net_t> * inputs[batch_size];
                    std::vector<float> * outputs[batch_size];
                    for(unsigned int i=0; i<batch_size; i++) {
                        inputs[i] = tasks[i].input;
                        outputs[i] = tasks[i].output;
                    }
                    
                    run_forward(inputs, outputs, batch_size);
                    for(unsigned int i=0; i<count; i++) {
                        tasks[i].prom.set_value();
                    }
                }
            });
    
            worker.detach();
        }
        m_workers_launched = true;
    }

    // to let the worker thread do it, push the network evaluation into the task queue
    // and signal conditional variable that something is pushed.
    std::future<void> ret;
    {
        std::lock_guard<std::mutex> lock(m_task_mutex);
        ForwardTask tsk(&input, &output);
        ret = tsk.prom.get_future();
        m_task_queue.emplace_back(std::move(tsk));
    }
    m_task_cond.notify_one();

    // this method will return when worker thread finishes its job
    ret.get();
#endif // USE_OPENCL_BATCHING
}

OpenCL_Network::OpenCL_Network() {
}
void OpenCL_Network::run_forward(const std::vector<net_t> ** inputs,
                             std::vector<float> ** outputs,
                             size_t batch_size) {
    constexpr auto width = 19;
    constexpr auto height = 19;
    constexpr auto one_plane = width * height * sizeof(net_t);

    opencl.ensure_thread_initialized();

    if (!opencl_thread_data.m_buffers_allocated) {
	auto iter = opencl_thread_data.m_convolve1_kernel.end();
	iter--;
        auto maxBatchSize = iter->first;
        auto maxInBufferSize = 0;
        auto maxMergeSize = 0;
        for (const auto& layer : m_layers) {
            auto channelGroups = layer.channels / (layer.channels % 8 ? 2 : 8);
            maxMergeSize = std::max<int>(maxMergeSize,
                                         layer.outputs * channelGroups);
            maxInBufferSize = std::max<int>(maxInBufferSize, layer.channels);
        }
        const auto alloc_inSize = one_plane *  maxInBufferSize * maxBatchSize;
        const auto alloc_mergeSize = one_plane * maxMergeSize * maxBatchSize;

        opencl_thread_data.m_inBuffer = cl::Buffer(
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_tmpBuffer = cl::Buffer(
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_residualBuffer = cl::Buffer(
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_thread_data.m_mergeBuffer = cl::Buffer(
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_mergeSize);
        opencl_thread_data.m_buffers_allocated = true;
    }

    cl::Buffer & inBuffer = opencl_thread_data.m_inBuffer;
    cl::Buffer & tmpBuffer = opencl_thread_data.m_tmpBuffer;
    cl::Buffer & mergeBuffer = opencl_thread_data.m_mergeBuffer;
    cl::Buffer & residualBuffer = opencl_thread_data.m_residualBuffer;
    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    // a null input pointer means we just want to test if the task runs
    if(inputs != nullptr) {
        std::vector<net_t> interleaved_input(inputs[0]->size() * batch_size);
        for(unsigned int j=0; j<batch_size; j++) {
            if(inputs[j] != nullptr) {
                for(size_t i=0; i<inputs[0]->size(); i++) {
                    interleaved_input[i * batch_size + j] = inputs[j]->at(i);
                }
            }
        }
    
        const auto inSize = sizeof(net_t) * inputs[0]->size() * batch_size;
        queue.enqueueWriteBuffer(inBuffer, CL_FALSE, 0, inSize, interleaved_input.data());
    }

    for (const auto& layer : m_layers) {
        if (layer.is_batchnorm) {
            auto bn_weights = begin(layer.weights);
            batchnorm(batch_size,
                      layer.outputs,
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
            const auto inBufferSize = layer.channels * one_plane * batch_size;
            queue.enqueueCopyBuffer(inBuffer, residualBuffer, 0, 0, inBufferSize);
            convolve(batch_size,
                     layer.filter_size,
                     layer.channels,
                     layer.outputs,
                     inBuffer,
                     tmpBuffer,
                     mergeBuffer,
                     conv1_weights);
            std::swap(inBuffer, tmpBuffer);
            batchnorm(batch_size,
                      layer.outputs,
                      361,
                      inBuffer,
                      tmpBuffer,
                      nullptr,
                      bn1_weights);
            std::swap(inBuffer, tmpBuffer);
            convolve(batch_size,
                     layer.filter_size,
                     layer.channels,
                     layer.outputs,
                     inBuffer,
                     tmpBuffer,
                     mergeBuffer,
                     conv2_weights);
            std::swap(inBuffer, tmpBuffer);
            batchnorm(batch_size,
                      layer.outputs,
                      361,
                      inBuffer,
                      tmpBuffer,
                      &residualBuffer,
                      bn2_weights);
            std::swap(inBuffer, tmpBuffer);
        } else  {
            auto conv_weights = begin(layer.weights);
            // plain convolution
            convolve(batch_size,
                     layer.filter_size,
                     layer.channels,
                     layer.outputs,
                     inBuffer,
                     tmpBuffer,
                     mergeBuffer,
                     conv_weights);
            std::swap(inBuffer, tmpBuffer);
        }
    }

    // a null output pointer means that we just want to test if the kernel runs.
    if(outputs != nullptr) {
        std::vector<net_t> interleaved_output(outputs[0]->size() * batch_size);
        const auto finalSize = m_layers.back().outputs * one_plane * batch_size;
        queue.enqueueReadBuffer(inBuffer, CL_FALSE, 0, finalSize, interleaved_output.data());
        queue.finish();

        for(unsigned int j=0; j<batch_size; j++) {
            if(outputs[j] != nullptr) {
                for(size_t i=0; i<outputs[0]->size(); i++) {
                    (*outputs[j])[i] = interleaved_output[batch_size * i + j];
                }
            }
        }
    } else {
        queue.finish();
    }
}

void OpenCL_Network::convolve(size_t batch_size, int filter_size, int channels, int outputs,
                              cl::Buffer& bufferInput,
                              cl::Buffer& bufferOutput,
                              cl::Buffer& bufferMerge,
                              weight_slice_t weights) {
    // fixed for 19x19
    constexpr int width = 19;
    constexpr int height = 19;
    constexpr int boardsize = width * height;

    cl::Kernel * m_convolve_kernel = nullptr;
    if (filter_size == 3) {
        m_convolve_kernel = &opencl_thread_data.m_convolve3_kernel[batch_size];
    } else {
        assert(filter_size == 1);
        m_convolve_kernel = &opencl_thread_data.m_convolve1_kernel[batch_size];
    }

    // Input channel grouping in multiples of 8
    int channelGroup = 8;
    int channelShift = 3;

    // Input layer is not a multiple of 8
    if (channels % 8 != 0) {
        assert(channels % 2 == 0);
        channelGroup = 2;
        channelShift = 1;
    }

    constexpr int rowGroup = 1;
    size_t outputGroup = std::min(outputs, 32);

#ifndef NDEBUG
    // Total output size after reducing
    size_t outSize = width * height * outputs * sizeof(net_t);

    // Produce channel * output planes and merge them at the end
    size_t mergeSize = (channels >> channelShift) * outSize;
    assert(mergeSize <= bufferMerge.getInfo<CL_MEM_SIZE>());
#endif

    // Copy the rows locally
    size_t stripSize;
    int rowTileSize;
    int rowTiles;
    if (filter_size == 3) {
        stripSize = filter_size * (width + (filter_size - 1)) * sizeof(float);
        rowTiles    =  cfg_rowtiles;
        rowTileSize =  (19 + rowTiles - 1) / rowTiles;
    } else {
        assert(filter_size == 1);
        stripSize = width * sizeof(float);
        rowTiles    = 19;
        rowTileSize =  1;
        assert(channelGroup == 8); // hardcoded in kernel
    }

    int rowBuffer = std::min<int>(channelGroup, 7);
    size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);

    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    try {
        m_convolve_kernel->setArg(0, bufferInput);
        m_convolve_kernel->setArg(1, bufferMerge);
        m_convolve_kernel->setArg(2, weights[0]);
        m_convolve_kernel->setArg(3, cl::Local(stripSize * channelGroup * rowGroup * batch_size));
        m_convolve_kernel->setArg(4, cl::Local(rowSize * batch_size));
        if (filter_size == 3) {
            m_convolve_kernel->setArg(5, rowTileSize);
            m_convolve_kernel->setArg(6, rowBuffer);
            m_convolve_kernel->setArg(7, channelGroup);
            m_convolve_kernel->setArg(8, channelShift);
        }

        queue.enqueueNDRangeKernel(*m_convolve_kernel, cl::NullRange,
                                   cl::NDRange(channels, outputs, rowTiles),
                                   cl::NDRange(channelGroup, outputGroup, rowGroup));
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve" << filter_size << ": " << e.what() << ": "
	        << e.err() << std::endl;
        throw;
    }

    cl::Kernel & merge_kernel = opencl_thread_data.m_merge_kernel[batch_size];
    assert(channels % (1 << channelShift) == 0);

    try {
        merge_kernel.setArg(0, bufferMerge);
        merge_kernel.setArg(1, bufferOutput);
        merge_kernel.setArg(2, weights[1]);
        merge_kernel.setArg(3, channels >> channelShift);

        queue.enqueueNDRangeKernel(merge_kernel, cl::NullRange,
                                   cl::NDRange(outputs, boardsize),
                                   cl::NDRange(std::min(8, outputs), 19));
    } catch (const cl::Error &e) {
        std::cerr << "Error in merge: " << e.what() << ": "
	        << e.err() << std::endl;
        throw;
    }
}

void OpenCL_Network::batchnorm(size_t batch_size,
                               int outputs,
                               int channel_size,
                               cl::Buffer& bufferInput,
                               cl::Buffer& bufferOutput,
                               cl::Buffer* bufferResidual,
                               weight_slice_t weights) {
    cl::CommandQueue & queue = opencl_thread_data.m_commandqueue;

    cl::Kernel & batchnorm_kernel = opencl_thread_data.m_batchnorm_kernel[batch_size];

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
    cl::Error last_error(0);
    for(size_t batch_size : {1,2,4,8}) {
        cl::Program p;
        try {
            p = cl::Program(sourceCode_config
                                    + sourceCode_convolve1
                                    + sourceCode_convolve3
                                    + sourceCode_utility);
        } catch (const cl::Error &e) {
            myprintf("Error getting kernels: %s: %d", e.what(), e.err());
            last_error = e;
            continue;
        }
        // Build program for these specific devices
        try {
    	    std::string args = "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";
#ifdef USE_HALF
            args += " -DUSE_HALF";
#endif
            args += " -DBATCH_SIZE=" + boost::lexical_cast<std::string>(batch_size);
            p.build(args.c_str());
        } catch (const cl::Error& e) {
            myprintf("Error building kernels: %s\n",
                        p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()).c_str());
            last_error = e;
            continue;
        }
        m_program[batch_size] = p;
    }
    if(m_program.empty()) {
        throw last_error;
    }

    ensure_thread_initialized();

    m_wavefront_size =
        opencl_thread_data.m_convolve3_kernel[1].getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
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
