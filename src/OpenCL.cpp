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

#ifdef USE_OPENCL
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "GTP.h"
#include "Network.h"
#include "OpenCL.h"
#include "Tuner.h"
#include "Utils.h"

using namespace Utils;

template <typename net_t> static std::string getClArgs();

template <>
std::string getClArgs<float>() {
    return "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros "
           "-cl-denorms-are-zero";
}
#ifdef USE_HALF
template <>
std::string getClArgs<half_float::half>() {
    return "-DUSE_HALF "
           "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros "
           "-cl-denorms-are-zero";
}
#endif

const std::string sourceCode_common =
    #include "kernels/common.opencl"
;

static const std::string sourceCode_tensorcore_test =
    #include "kernels/tensorcore_test.opencl"
;

static const std::string sourceCode_config = R"(
#define BOARD_SIZE )" + std::to_string(BOARD_SIZE) +
"\n#define NUM_INTERSECTIONS " + std::to_string(NUM_INTERSECTIONS) +
"\n#define WINOGRAD_M " + std::to_string(WINOGRAD_M) +
"\n#define WINOGRAD_ALPHA " + std::to_string(WINOGRAD_ALPHA) +
"\n#define WTILES " + std::to_string(WINOGRAD_WTILES);

static const std::string sourceCode_convolve1 =
    #include "kernels/convolve1.opencl"
;

static const std::string sourceCode_convolve3 =
    #include "kernels/convolve3.opencl"
;

static std::string sourceCode_global_avg_pooling =
    #include "kernels/pooling.opencl"
;

static std::string sourceCode_apply_se =
    #include "kernels/apply_se.opencl"
;

const std::string sourceCode_sgemm =
    "#if TCE == 1\n" // Enable tensorcore
    #include "kernels/clblast/hgemm_tensorcore.opencl"
    "\n#else\n" // Use clblast
    #include "kernels/clblast/xgemm_part1.opencl"
    #include "kernels/clblast/xgemm_part2.opencl"
    #include "kernels/clblast/xgemm_part3.opencl"
    #include "kernels/clblast/xgemm_batched.opencl"
    "\n#endif\n"
;

const std::string sourceCode_sgemv =
    #include "kernels/clblast/xgemv.opencl"
;


template <typename net_t>
void OpenCL<net_t>::ensure_context_initialized(OpenCLContext& opencl_context) {
    if (!opencl_context.m_is_initialized) {
        // Make kernels
        opencl_context.m_convolve1_kernel =
            cl::Kernel(m_program, "convolve1");
        opencl_context.m_merge_kernel =
            cl::Kernel(m_program, "merge");
        opencl_context.m_in_transform_kernel =
            cl::Kernel(m_program, "in_transform");
        opencl_context.m_sgemm_kernel =
            cl::Kernel(m_program, "XgemmBatched");
        opencl_context.m_out_transform_bn_kernel =
            cl::Kernel(m_program, "out_transform_fused_bn");
        opencl_context.m_sgemv_kernel =
            cl::Kernel(m_program, "Xgemv");
        opencl_context.m_out_transform_bn_in_kernel =
            cl::Kernel(m_program, "out_transform_fused_bn_in");
        opencl_context.m_commandqueue = cl::CommandQueue(m_context, m_device);
        opencl_context.m_global_avg_pooling_kernel =
            cl::Kernel(m_program, "global_avg_pooling");
        opencl_context.m_apply_se_kernel =
            cl::Kernel(m_program, "apply_se");
        opencl_context.m_is_initialized = true;
    }
}

template <typename net_t>
void OpenCL_Network<net_t>::add_weights(const size_t layer, const size_t size,
                                        const net_t* const weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(Layer());
    }

    auto weightSize = size * sizeof(net_t);

    auto queue = cl::CommandQueue(getOpenCL().m_context, getOpenCL().m_device);
    auto buffer =
        cl::Buffer(m_opencl.m_context, CL_MEM_READ_ONLY, weightSize, nullptr);
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, weightSize,
                             const_cast<net_t*>(weights));
    m_layers.back().weights.push_back(std::move(buffer));
}

template <typename net_t>
void OpenCL_Network<net_t>::forward(const std::vector<float>& input,
                                    std::vector<float>& output_pol,
                                    std::vector<float>& output_val,
                                    OpenCLContext& opencl_context,
                                    const int batch_size) {
    constexpr auto tiles = WINOGRAD_P;
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(net_t);
    const auto finalSize_pol =
        m_layers[m_layers.size() - 2].outputs * one_plane;
    const auto finalSize_val = m_layers.back().outputs * one_plane;

    m_opencl.ensure_context_initialized(opencl_context);

    if (!opencl_context.m_buffers_allocated) {
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_channels =
                std::max(max_channels, std::max(layer.channels, layer.outputs));
        }

        const auto mwg = m_opencl.m_sgemm_tuners.mwg;
        const auto nwg = m_opencl.m_sgemm_tuners.nwg;
        const auto vwm = m_opencl.m_sgemm_tuners.vwm;
        const auto vwn = m_opencl.m_sgemm_tuners.vwn;

        const auto m_ceil = ceilMultiple(ceilMultiple(max_channels, mwg), vwm);
        const auto n_ceil = ceilMultiple(ceilMultiple(tiles, nwg), vwn);

        const auto alloc_inSize = getOpenCL().m_batch_size * NUM_INTERSECTIONS
                                  * max_channels * sizeof(net_t);
        const auto alloc_pool_size =
            getOpenCL().m_batch_size * 2 * max_channels * sizeof(net_t);
        const auto alloc_vm_size = getOpenCL().m_batch_size * WINOGRAD_TILE
                                   * m_ceil * n_ceil * sizeof(net_t);

        auto v_zeros = std::vector<net_t>(alloc_vm_size);

        opencl_context.m_inBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);

        opencl_context.m_inBuffer2 = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);

        // Zero pad the unused areas in V.
        // Zeros must not be overwritten or convolution gives incorrect results.
        opencl_context.m_VBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
            alloc_vm_size, v_zeros.data(), nullptr);

        opencl_context.m_MBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_vm_size);

        opencl_context.m_pinnedOutBuffer_pol = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            getOpenCL().m_batch_size * finalSize_pol);
        opencl_context.m_pinnedOutBuffer_val = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            getOpenCL().m_batch_size * finalSize_val);

        opencl_context.m_pool_buffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_pool_size);

        opencl_context.m_buffers_allocated = true;
    }

    cl::Buffer& inBuffer = opencl_context.m_inBuffer;
    cl::Buffer& inBuffer2 = opencl_context.m_inBuffer2;
    cl::Buffer& VBuffer = opencl_context.m_VBuffer;
    cl::Buffer& MBuffer = opencl_context.m_MBuffer;
    cl::CommandQueue& queue = opencl_context.m_commandqueue;

    std::vector<net_t> net_t_input(input.size());
    std::copy(begin(input), end(input), begin(net_t_input));
    cl::Buffer& pool_buffer = opencl_context.m_pool_buffer;

    const auto inSize = sizeof(net_t) * input.size();
    queue.enqueueWriteBuffer(inBuffer, CL_FALSE, 0, inSize, net_t_input.data());

    // Fused in_out transformation kernel is slower with big batch_sizes than
    // calling out and in transformations separately.
    // This condition could be tunable in future.
    auto use_inout = (batch_size == 1);

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
                skip_next_in_trans = use_inout;
            }

            convolve3(opencl_context,
                      layer.channels,
                      layer.outputs,
                      inBuffer,
                      inBuffer,
                      VBuffer,
                      MBuffer,
                      conv_weights,
                      nullptr,
                      bn_weights,
                      skip_in_trans, skip_next_in_trans, true,
                      true,
                      batch_size);

            skip_in_trans = skip_next_in_trans;
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != cend(m_layers));

            auto conv1_weights = begin(layer.weights);
            auto   bn1_weights = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 3;
            auto   bn2_weights = begin(layer.weights) + 4;

            convolve3(opencl_context,
                      layer.channels,
                      layer.outputs,
                      inBuffer,
                      inBuffer2,
                      VBuffer,
                      MBuffer,
                      conv1_weights,
                      nullptr,
                      bn1_weights,
                      skip_in_trans, use_inout, false,
                      true,
                      batch_size);

            auto skip_next_in_trans = false;
            if (niter->is_residual_block) {
                skip_next_in_trans = use_inout;
            }

            if (niter->is_se_unit) {
                // Residual connection and relu are applied in SE-unit.
                convolve3(opencl_context,
                          layer.channels,
                          layer.outputs,
                          inBuffer2,
                          inBuffer2,
                          VBuffer,
                          MBuffer,
                          conv2_weights,
                          nullptr,
                          bn2_weights,
                          use_inout, skip_next_in_trans, false,
                          false,
                          batch_size);
            } else {
                 convolve3(opencl_context,
                           layer.channels,
                           layer.outputs,
                           inBuffer2,
                           inBuffer,
                           VBuffer,
                           MBuffer,
                           conv2_weights,
                           &inBuffer,
                           bn2_weights,
                           use_inout, skip_next_in_trans, true,
                           true,
                           batch_size);
            }

            skip_in_trans = skip_next_in_trans;
        } else if (layer.is_se_unit) {
            assert(layer.channels == layer.outputs);
            assert(niter != cend(m_layers));
            auto se_weights = begin(layer.weights);

            squeeze_excitation(opencl_context,
                               layer.channels,
                               layer.outputs,
                               inBuffer2,
                               pool_buffer,
                               MBuffer,
                               se_weights,
                               inBuffer,
                               batch_size);
        } else {
            assert(layer.is_convolve1);

            cl::Buffer out_buffer;
            if (niter == cend(m_layers)) {
                out_buffer = opencl_context.m_pinnedOutBuffer_val;
            } else {
                out_buffer = opencl_context.m_pinnedOutBuffer_pol;
            }

            convolve1(opencl_context, layer.channels,
                      layer.outputs,
                      inBuffer,
                      out_buffer,
                      VBuffer,
                      begin(layer.weights),
                      batch_size);
        }
    }

    auto pinnedOutBufferHost_pol =
        queue.enqueueMapBuffer(opencl_context.m_pinnedOutBuffer_pol, CL_FALSE,
                               CL_MAP_READ, 0, batch_size * finalSize_pol);
    auto pinnedOutBufferHost_val =
        queue.enqueueMapBuffer(opencl_context.m_pinnedOutBuffer_val, CL_FALSE,
                               CL_MAP_READ, 0, batch_size * finalSize_val);

    {
        // Finish call is usually a busy wait. When using multiple threads
        // use the lock to avoid busy waiting with all threads.
        std::lock_guard<std::mutex> lock(m_queue_finish_mutex);
        queue.finish();
    }

    auto polptr = static_cast<net_t*>(pinnedOutBufferHost_pol);
    auto valptr = static_cast<net_t*>(pinnedOutBufferHost_val);
    std::copy(polptr, polptr + output_pol.size(), begin(output_pol));
    std::copy(valptr, valptr + output_val.size(), begin(output_val));

    queue.enqueueUnmapMemObject(opencl_context.m_pinnedOutBuffer_pol,
                                pinnedOutBufferHost_pol);
    queue.enqueueUnmapMemObject(opencl_context.m_pinnedOutBuffer_val,
                                pinnedOutBufferHost_val);
}

template <typename net_t>
void OpenCL_Network<net_t>::squeeze_excitation(OpenCLContext& opencl_context,
                                               const int channels,
                                               const int fc_outputs,
                                               cl::Buffer& bufferIn,
                                               cl::Buffer& bufferTemp1,
                                               cl::Buffer& bufferTemp2,
                                               const weight_slice_t weights,
                                               cl::Buffer& bufferResidual,
                                               const int batch_size) {

    cl::Kernel& pooling_kernel = opencl_context.m_global_avg_pooling_kernel;
    cl::Kernel& apply_se_kernel = opencl_context.m_apply_se_kernel;
    cl::CommandQueue& queue = opencl_context.m_commandqueue;

    try {
        pooling_kernel.setArg(0, batch_size * channels);
        pooling_kernel.setArg(1, bufferIn);
        pooling_kernel.setArg(2, bufferTemp1);

        queue.enqueueNDRangeKernel(
            pooling_kernel, cl::NullRange,
            cl::NDRange(BOARD_SIZE, batch_size * channels),
            cl::NDRange(BOARD_SIZE, 1));
    } catch (const cl::Error& e) {
        std::cerr << "Error in squeeze_excitation: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }

    innerproduct(opencl_context, bufferTemp1, weights[0], weights[1],
                 bufferTemp2, channels, fc_outputs, true, batch_size);

    innerproduct(opencl_context, bufferTemp2, weights[2], weights[3],
                 bufferTemp1, fc_outputs, 2 * channels, false, batch_size);

    try {
        apply_se_kernel.setArg(0, channels);
        apply_se_kernel.setArg(1, batch_size);
        apply_se_kernel.setArg(2, bufferIn);
        apply_se_kernel.setArg(3, bufferResidual);
        apply_se_kernel.setArg(4, bufferTemp1);

        queue.enqueueNDRangeKernel(
            apply_se_kernel, cl::NullRange,
            cl::NDRange(BOARD_SIZE, batch_size * channels));
    } catch (const cl::Error& e) {
        std::cerr << "Error in squeeze_excitation: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }
}

template <typename net_t>
void OpenCL_Network<net_t>::innerproduct(OpenCLContext& opencl_context,
                                         const cl::Buffer& input,
                                         const cl::Buffer& weights,
                                         const cl::Buffer& biases,
                                         cl::Buffer& output,
                                         const int inputs, const int outputs,
                                         const bool relu,
                                         const int batch_size) {

    auto sgemv_kernel = opencl_context.m_sgemv_kernel;
    cl::CommandQueue& queue = opencl_context.m_commandqueue;

    // TODO: Tune these
    const auto wgs1 = size_t{32};
    const auto wpt1 = size_t{1};

    const auto m_ceil = int(ceilMultiple(outputs, wgs1 * wpt1));
    const auto global_size = m_ceil / wpt1;
    const auto local_size = wgs1;

    try {
        // Sets the kernel arguments
        sgemv_kernel.setArg(0, outputs);
        sgemv_kernel.setArg(1, inputs);
        sgemv_kernel.setArg(2, weights);
        sgemv_kernel.setArg(3, 0);
        sgemv_kernel.setArg(4, inputs);
        sgemv_kernel.setArg(5, input);
        sgemv_kernel.setArg(6, 0);
        sgemv_kernel.setArg(7, output);
        sgemv_kernel.setArg(8, 0);
        sgemv_kernel.setArg(9, biases);
        sgemv_kernel.setArg(10, static_cast<int>(relu));

        queue.enqueueNDRangeKernel(sgemv_kernel, cl::NullRange,
                                   cl::NDRange(global_size, batch_size),
                                   cl::NDRange(local_size, 1));
    } catch (const cl::Error& e) {
        std::cerr << "Error in innerproduct: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }
}


template <typename net_t>
void OpenCL_Network<net_t>::convolve3(OpenCLContext& opencl_context,
                                      const int channels, const int outputs,
                                      cl::Buffer& bufferIn,
                                      cl::Buffer& bufferOut,
                                      cl::Buffer& bufferV,
                                      cl::Buffer& bufferM,
                                      const weight_slice_t weights,
                                      cl::Buffer* const bufferResidual,
                                      const weight_slice_t bn_weights,
                                      const bool skip_in_transform,
                                      const bool fuse_in_transform,
                                      const bool store_inout,
                                      const bool relu,
                                      const int batch_size) {

    cl::Kernel& in_transform_kernel = opencl_context.m_in_transform_kernel;
    cl::Kernel& sgemm_kernel = opencl_context.m_sgemm_kernel;
    cl::Kernel& out_transform_bn_kernel =
        opencl_context.m_out_transform_bn_kernel;
    cl::Kernel& out_transform_bn_in_kernel =
        opencl_context.m_out_transform_bn_in_kernel;

    auto mwg = m_opencl.m_sgemm_tuners.mwg;
    auto nwg = m_opencl.m_sgemm_tuners.nwg;
    auto kwg = m_opencl.m_sgemm_tuners.kwg;
    auto vwm = m_opencl.m_sgemm_tuners.vwm;
    auto vwn = m_opencl.m_sgemm_tuners.vwn;
    auto mdimc = m_opencl.m_sgemm_tuners.mdimc;
    auto ndimc = m_opencl.m_sgemm_tuners.ndimc;
    auto tce = m_opencl.m_sgemm_tuners.tce;
    auto mdima = m_opencl.m_sgemm_tuners.mdima;
    auto ndimb = m_opencl.m_sgemm_tuners.ndimb;

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

    auto wgs = ceilMultiple(batch_size * tiles, wavefront_size);
    auto wgs_single = ceilMultiple(tiles, wavefront_size);

    auto m_ceil = int(ceilMultiple(ceilMultiple(outputs, mwg), vwm));
    auto n_ceil = int(ceilMultiple(ceilMultiple(batch_size * tiles, nwg), vwn));
    auto k_ceil = int(ceilMultiple(ceilMultiple(channels, kwg), vwm));

    cl::CommandQueue& queue = opencl_context.m_commandqueue;

    if (!skip_in_transform) {
        try {
            in_transform_kernel.setArg(0, bufferIn);
            in_transform_kernel.setArg(1, bufferV);
            in_transform_kernel.setArg(2, channels);
            in_transform_kernel.setArg(3, k_ceil);
            in_transform_kernel.setArg(4, n_ceil);
            in_transform_kernel.setArg(5, batch_size);

            // No relu not implemented
            assert(relu);

            queue.enqueueNDRangeKernel(in_transform_kernel, cl::NullRange,
                                       cl::NDRange(wgs, channels));
        } catch (const cl::Error& e) {
            std::cerr << "Error in convolve3/in: " << e.what() << ": "
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
                                  cl::size_type(WINOGRAD_TILE)};

        // tensorcore implementation uses a different dimension
        if (tce) {
            local_sgemm = {32 * mdimc / mdima, ndimc / ndimb, 1};
            size_sgemm = {32 * m_ceil / mdima * mdimc / mwg,
                          n_ceil / ndimb * ndimc / nwg,
                          cl::size_type(WINOGRAD_TILE)};
        }
        queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange,
                                   size_sgemm, local_sgemm);
    } catch (const cl::Error& e) {
        std::cerr << "Error in convolve3/sgemm: " << e.what() << ": " << e.err()
                  << std::endl;
        throw;
    }

    try {
        if (fuse_in_transform) {
            assert(relu);
            // TODO : Eventually this might also be something tuneable?
            // Needs to match OUTIN_KWG in kernel
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

            queue.enqueueNDRangeKernel(
                out_transform_bn_in_kernel, cl::NullRange,
                cl::NDRange(outputs, wgs_single, batch_size),
                cl::NDRange(dim_size, wgs_single, 1));
        } else {
            out_transform_bn_kernel.setArg(0, bufferM);
            out_transform_bn_kernel.setArg(1, bufferOut);
            out_transform_bn_kernel.setArg(2, outputs);
            out_transform_bn_kernel.setArg(3, m_ceil);
            out_transform_bn_kernel.setArg(4, n_ceil);
            out_transform_bn_kernel.setArg(5, batch_size);
            if (bufferResidual) {
                out_transform_bn_kernel.setArg(6, *bufferResidual);
            } else {
                out_transform_bn_kernel.setArg(6, nullptr);
            }
            out_transform_bn_kernel.setArg(7, bn_weights[0]);
            out_transform_bn_kernel.setArg(8, bn_weights[1]);
            out_transform_bn_kernel.setArg(9, static_cast<int>(relu));

            // Needs to match OUT_KWG, OUT_BWG in the kernel.
            // This could be tuned.
            cl::NDRange local_out = {32, 2};

            cl::NDRange global_out = {
                ceilMultiple(outputs, local_out[0]),
                ceilMultiple(tiles * batch_size, local_out[1])};

            queue.enqueueNDRangeKernel(out_transform_bn_kernel, cl::NullRange,
                                       global_out, local_out);
        }
    } catch (const cl::Error& e) {
        std::cerr << "Error in convolve3/out: " << e.what() << ": " << e.err()
                  << std::endl;
        throw;
    }
}

template <typename net_t>
void OpenCL_Network<net_t>::convolve1(OpenCLContext& opencl_context,
                                      const int channels, const int outputs,
                                      cl::Buffer& bufferInput,
                                      cl::Buffer& bufferOutput,
                                      cl::Buffer& bufferMerge,
                                      const weight_slice_t weights,
                                      const int batch_size) {
    // The size of the board is defined at compile time
    constexpr int width = BOARD_SIZE;
    constexpr int boardsize = NUM_INTERSECTIONS;
    constexpr int rowTiles = BOARD_SIZE;

    // Input channel grouping in multiples of 8
    constexpr int channelGroup = 8;
    constexpr int channelShift = 3;
    constexpr int rowGroup = 1;
    size_t outputGroup = std::min(outputs, 32);

    auto m_convolve_kernel = &opencl_context.m_convolve1_kernel;

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

    cl::CommandQueue& queue = opencl_context.m_commandqueue;

    try {
        m_convolve_kernel->setArg(0, bufferInput);
        m_convolve_kernel->setArg(1, bufferMerge);
        m_convolve_kernel->setArg(2, weights[0]);
        m_convolve_kernel->setArg(
            3, cl::Local(stripSize * channelGroup * rowGroup));
        m_convolve_kernel->setArg(4, cl::Local(rowSize));

        queue.enqueueNDRangeKernel(
            *m_convolve_kernel, cl::NullRange,
            cl::NDRange(channels, outputs, batch_size * rowTiles),
            cl::NDRange(channelGroup, outputGroup, rowGroup));
    } catch (const cl::Error& e) {
        std::cerr << "Error in convolve1: " << e.what() << ": " << e.err()
                  << std::endl;
        throw;
    }

    cl::Kernel& merge_kernel = opencl_context.m_merge_kernel;
    assert(channels % (1 << channelShift) == 0);

    try {
        merge_kernel.setArg(0, bufferMerge);
        merge_kernel.setArg(1, bufferOutput);
        merge_kernel.setArg(2, channels >> channelShift);

        queue.enqueueNDRangeKernel(
            merge_kernel, cl::NullRange,
            cl::NDRange(outputs, boardsize, batch_size),
            cl::NDRange(std::min(8, outputs), BOARD_SIZE, 1));
    } catch (const cl::Error& e) {
        std::cerr << "Error in merge: " << e.what() << ": " << e.err()
                  << std::endl;
        throw;
    }
}

template <class T>
static std::string opencl_dev_type_to_string(const T type) {
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

template <typename net_t>
void OpenCL<net_t>::process_tuners(std::string tuners) {
    std::string buf;
    std::stringstream ss(tuners);
    std::size_t found;

    auto mwg = false;
    auto nwg = false;
    auto kwg = false;
    auto ndimc = false;
    auto mdimc = false;
    auto mdima = false;
    auto ndimb = false;
    auto vwm = false;
    auto vwn = false;
    auto tce = false;

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
        if (name == "-DMDIMA") {
            m_sgemm_tuners.mdima = value;
            mdima = true;
        }
        if (name == "-DNDIMB") {
            m_sgemm_tuners.ndimb = value;
            ndimb = true;
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
        if (name == "-DTCE") {
            m_sgemm_tuners.tce = value;
            tce = true;
        }
    }
    if (!mwg || !nwg || !kwg || !mdimc || !ndimc
        || !vwm || !vwn || !mdima || !ndimb) {
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
        if (!mdima) {
            std::cerr << " MDIMA";
        }
        if (!ndimb) {
            std::cerr << " NDIMB";
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
        if (!tce) {
            std::cerr << " VWN";
        }
        std::cerr << std::endl;
        std::exit(-1);
    }
}

template <typename net_t>
std::vector<size_t> OpenCL<net_t>::get_sgemm_tuners() {
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

template <typename net_t>
OpenCL<net_t>::OpenCL(const int gpu, const bool silent) {
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error& e) {
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

    for (const auto& p : platforms) {
        std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
        if (!silent) {
            std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
            std::string platname = p.getInfo<CL_PLATFORM_NAME>();
            std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
            myprintf("Platform version: %s\n", platvers.c_str());
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
        } catch (const cl::Error& e) {
            myprintf("Error getting device(s): %s: %d\n", e.what(), e.err());
            devices.clear();
        }
        for (auto& d : devices) {
            if (!silent) {
                myprintf("Device ID:     %d\n", id);
                myprintf("Device name:   %s\n",
                         trim(d.getInfo<CL_DEVICE_NAME>()).c_str());
                myprintf("Device type:   %s\n",
                         opencl_dev_type_to_string(d.getInfo<CL_DEVICE_TYPE>())
                             .c_str());
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
            this_score +=
                1000 * boost::icontains(this_vendor, "advanced micro devices");
            this_score += 1000 * boost::icontains(this_vendor, "amd");
            this_score += 1000 * boost::icontains(this_vendor, "nvidia");
            this_score += 500 * boost::icontains(this_vendor, "intel");
            this_score +=
                100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
            this_score += opencl_version * 10;
            if (!silent) {
                myprintf("Device score:  %d\n", this_score);
            }

            bool preferred = (gpu == id);

            if (((this_score > best_score)
                 && (d.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_CPU))
                || preferred) {
                best_version = opencl_version;
                best_platform = p;
                best_device = d;
                best_vendor = this_vendor;
                if (preferred) {
                    best_score =
                        std::numeric_limits<decltype(best_score)>::max();
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
    } catch (const cl::Error& e) {
        myprintf("Error creating OpenCL context: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error creating OpenCL context.");
    }
    m_context = context;
    m_device = best_device;

    m_cl_args = getClArgs<net_t>();

    myprintf("Half precision compute support: ");
    if (m_device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp16")
        != std::string::npos) {
        myprintf("Yes.\n");
        m_fp16_compute = true;
        m_cl_args += " -DFP16_SUPPORT";
    } else {
        myprintf("No.\n");
    }

    myprintf("Tensor Core support: ");
    {
        // if this is a nvidia GPU, test-compile a sample inline assembly code
        // with tensor wmma instructions. if not, don't bother trying
        std::string this_vendor = m_device.getInfo<CL_DEVICE_VENDOR>();
        if (boost::icontains(this_vendor, "nvidia")) {
            try {
                cl::Program(m_context, sourceCode_tensorcore_test)
                    .build(m_cl_args.c_str());
                m_tensorcore = true;
                myprintf("Yes.\n");
            } catch (...) {
                myprintf("No.\n");
            }
        } else {
            myprintf("No.\n");
        }
    }
}

template <typename net_t>
void OpenCL<net_t>::initialize(const int channels, const size_t batch_size) {
    m_batch_size = batch_size;
    // Make program of the source code in the context
    try {
        m_program = cl::Program(
            m_context, sourceCode_common + sourceCode_config
                           + sourceCode_convolve1 + sourceCode_convolve3
                           + sourceCode_sgemm + sourceCode_global_avg_pooling
                           + sourceCode_sgemv + sourceCode_apply_se);
    } catch (const cl::Error& e) {
        myprintf("Error getting kernels: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error getting OpenCL kernels.");
    }

    auto t = Tuner<net_t>(*this, m_context, m_device);
    if (m_tensorcore) {
        t.enable_tensorcore();
    }

    auto sgemm_tuners = t.load_sgemm_tuners(channels, batch_size * WINOGRAD_P,
                                            channels, WINOGRAD_TILE);

    // Some NVIDIA drivers are buggy and will fail to compile the rest of the
    // kernels after a tuning run.
    if (cfg_tune_only) {
        // Originally this was an exit() but this will make the tuner
        // only tune the first GPU.  Return instead.  Exit will be called
        // after all GPUs are created.
        return;
    }

    // Build program for these specific devices
    try {
        std::string args = m_cl_args;
        // Intel iGPUs need vector types for math for best performance
        if (m_device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() > 1) {
            args += " -DWINOGRAD_SIMD";
        }

        args += sgemm_tuners;
        m_program.build(args.c_str());
    } catch (const cl::Error&) {
        myprintf(
            "Error building kernels: %s\n",
            m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device).c_str());
        throw std::runtime_error("Error building OpenCL kernels.");
    }

    OpenCLContext tdata;
    ensure_context_initialized(tdata);

    process_tuners(sgemm_tuners);

    m_wavefront_size =
        tdata.m_sgemm_kernel
            .getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
                m_device);
    myprintf("Wavefront/Warp size: %d\n", m_wavefront_size);

    m_max_workgroup_size = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    m_max_workgroup_dims = m_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    myprintf("Max workgroup size: %d\n", m_max_workgroup_size);
    myprintf("Max workgroup dimensions: ");
    for (auto d : m_max_workgroup_dims) {
        myprintf("%d ", d);
    }
    myprintf("\n");

    m_init_ok = true;
}

template <typename net_t>
bool OpenCL<net_t>::has_fp16_compute() {
    return m_fp16_compute;
}

template <typename net_t>
bool OpenCL<net_t>::has_tensor_cores() {
    return m_tensorcore;
}

template <typename net_t>
std::string OpenCL<net_t>::get_device_name() {
    std::stringstream ss;

    ss << "OpenCL: ";
    ss << m_device.getInfo<CL_DEVICE_VENDOR>() << " ";
    ss << m_device.getInfo<CL_DEVICE_NAME>() << " @ ";
    ss << m_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

    return ss.str();
}

template class OpenCL<float>;
template class OpenCL_Network<float>;
#ifdef USE_HALF
template class OpenCL<half_float::half>;
template class OpenCL_Network<half_float::half>;
#endif

#endif
