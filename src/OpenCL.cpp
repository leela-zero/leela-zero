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

#ifdef USE_OPENCL
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

#include "OpenCL.h"
#include "Network.h"
#include "GTP.h"
#include "Utils.h"
#include "cuda_kernels.h"

using namespace Utils;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define ReportCUBLASErrors(status) CublasError((status), __FILE__, TOSTRING(__LINE__))
#define ReportCUDAErrors(status) CudaError((status), __FILE__, TOSTRING(__LINE__))

static const std::string sourceCode_config = R"(
#define BOARD_SIZE )" + std::to_string(BOARD_SIZE) +
"\n#define NUM_INTERSECTIONS " + std::to_string(NUM_INTERSECTIONS) +
"\n#define WINOGRAD_M " + std::to_string(WINOGRAD_M) +
"\n#define WINOGRAD_ALPHA " + std::to_string(WINOGRAD_ALPHA) +
"\n#define WTILES " + std::to_string(WINOGRAD_WTILES);

template <typename net_t>
void OpenCL<net_t>::ensure_context_initialized(OpenCLContext &opencl_context) {
    if (!opencl_context.m_is_initialized) {
		ReportCUBLASErrors(cublasCreate(&opencl_context.m_cublas));
        opencl_context.m_is_initialized = true;
    }
}

template <typename net_t>
void OpenCL_Network<net_t>::add_weights(size_t layer,
                                 size_t size,
                                 const net_t * weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(Layer());
    }

    auto weightSize = size * sizeof(net_t);

    void *device_mem;
    ReportCUDAErrors(cudaMalloc((void**)&device_mem, weightSize));
    ReportCUDAErrors(cudaMemcpyAsync(device_mem, (net_t*)&weights[0], weightSize,
                cudaMemcpyHostToDevice));
    m_layers.back().weights.emplace_back(device_mem);
}

template <typename net_t>
void OpenCL_Network<net_t>::forward(const std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val,
                             OpenCLContext & opencl_context,
                             const int batch_size) {
    constexpr auto tiles = WINOGRAD_P;
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(net_t);
    const auto finalSize_pol = m_layers[m_layers.size()-2].outputs * one_plane;
    const auto finalSize_val = m_layers.back().outputs * one_plane;

    m_opencl.ensure_context_initialized(opencl_context);

    if (!opencl_context.m_buffers_allocated) {
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
            MAX_BATCH * NUM_INTERSECTIONS * max_channels * sizeof(net_t);
        const auto alloc_vm_size =
            MAX_BATCH * WINOGRAD_TILE * m_ceil * n_ceil * sizeof(net_t);

        auto v_zeros = std::vector<net_t>(alloc_vm_size);

        ReportCUDAErrors(cudaMalloc((void**)&opencl_context.m_inBuffer, alloc_inSize));
        ReportCUDAErrors(cudaMalloc((void**)&opencl_context.m_inBuffer2, alloc_inSize));
        ReportCUDAErrors(cudaMalloc((void**)&opencl_context.m_VBuffer, alloc_vm_size));

        // Zero initialize VBuffer
        ReportCUDAErrors(cudaMemcpy((void*)opencl_context.m_VBuffer,
                  (net_t*)&v_zeros.data()[0], alloc_vm_size, cudaMemcpyHostToDevice));

        ReportCUDAErrors(cudaMalloc((void**)&opencl_context.m_MBuffer, alloc_vm_size));

        ReportCUDAErrors(cudaMalloc((void**)&opencl_context.m_pinnedOutBuffer_pol,
                    MAX_BATCH * finalSize_pol));

        ReportCUDAErrors(cudaMalloc((void**)&opencl_context.m_pinnedOutBuffer_val,
                    MAX_BATCH * finalSize_val));

        opencl_context.m_buffers_allocated = true;
    }

    void* &inBuffer = opencl_context.m_inBuffer;
    void* &inBuffer2 = opencl_context.m_inBuffer2;
    void* &VBuffer = opencl_context.m_VBuffer;
    void* &MBuffer = opencl_context.m_MBuffer;
    //cl::CommandQueue &queue = opencl_context.m_commandqueue;

    std::vector<net_t> net_t_input(input.size());
    std::copy(begin(input), end(input), begin(net_t_input));

    const auto inSize = sizeof(net_t) * input.size();
    ReportCUDAErrors(cudaMemcpy((void**)inBuffer,
              (net_t*)&net_t_input.data()[0], inSize, cudaMemcpyHostToDevice));

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
                     batch_size);

            skip_in_trans = skip_next_in_trans;
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != cend(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto bn1_weights   = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 3;
            auto bn2_weights   = begin(layer.weights) + 4;
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
                      skip_in_trans, true, false,
                      batch_size);

            auto skip_next_in_trans = false;
            if (niter->is_residual_block) {
                skip_next_in_trans = true;
            }
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
                      true, skip_next_in_trans, true,
                      batch_size);
            skip_in_trans = skip_next_in_trans;
        } else {
            assert(layer.is_convolve1);

			//auto outputs = layer.channels;
			//auto out_size_temp = 20;
			//std::vector<float> out_temp(out_size_temp);

			//ReportCUDAErrors(cudaMemcpy(&out_temp.data()[0], inBuffer,
			//		  out_size_temp * sizeof(float), cudaMemcpyDeviceToHost));

			//for (int i = 0; i < out_size_temp; i++) {
			//	myprintf("%f ", out_temp[i]);
			//}
			//myprintf("\n\n");

            void* out_buffer;
            void* host_out_buffer;
			size_t out_size;
			// Assumes that value head was pushed last
            if (niter == cend(m_layers)) {
                host_out_buffer = &output_val.data()[0];
                out_buffer = opencl_context.m_pinnedOutBuffer_val;
				out_size = batch_size * finalSize_val;
            } else {
                host_out_buffer = &output_pol.data()[0];
                out_buffer = opencl_context.m_pinnedOutBuffer_pol;
				out_size = batch_size * finalSize_pol;
            }

            convolve1(opencl_context, layer.channels,
                    layer.outputs,
                    inBuffer,
                    out_buffer,
                    VBuffer,
                    begin(layer.weights),
                    batch_size);

            ReportCUDAErrors(cudaMemcpyAsync(host_out_buffer, out_buffer,
                      out_size, cudaMemcpyDeviceToHost));
        }
    }

    //auto pinnedOutBufferHost_pol = queue.enqueueMapBuffer(
    //    opencl_context.m_pinnedOutBuffer_pol, CL_FALSE,
    //    CL_MAP_READ, 0, batch_size * finalSize_pol);
    //auto pinnedOutBufferHost_val = queue.enqueueMapBuffer(
    //    opencl_context.m_pinnedOutBuffer_val, CL_FALSE,
    //    CL_MAP_READ, 0, batch_size * finalSize_val);

    {
        // Finish call is usually a busy wait. When using multiple threads
        // use the lock to avoid busy waiting with all threads.
        std::lock_guard<std::mutex> lock(m_queue_finish_mutex);
		//TODO: wait for stream instead?
	    cudaDeviceSynchronize();
    }

    //auto polptr = static_cast<net_t*>(pinnedOutBufferHost_pol);
    //auto valptr = static_cast<net_t*>(pinnedOutBufferHost_val);
    //std::copy(polptr, polptr + output_pol.size(), begin(output_pol));
    //std::copy(valptr, valptr + output_val.size(), begin(output_val));

    //queue.enqueueUnmapMemObject(opencl_context.m_pinnedOutBuffer_pol,
    //        pinnedOutBufferHost_pol);
    //queue.enqueueUnmapMemObject(opencl_context.m_pinnedOutBuffer_val,
    //        pinnedOutBufferHost_val);

}

template <typename net_t>
void OpenCL_Network<net_t>::convolve3(OpenCLContext & opencl_context,
                              int channels, int outputs,
                              void* bufferIn,
                              void* bufferOut,
                              void* bufferV,
                              void* bufferM,
                              weight_slice_t weights,
                              void** bufferResidual,
                              weight_slice_t bn_weights,
                              bool skip_in_transform,
                              bool fuse_in_transform,
                              bool store_inout,
                              int batch_size) {

 //   cl::Kernel & in_transform_kernel = opencl_context.m_in_transform_kernel;
 //   cl::Kernel & sgemm_kernel = opencl_context.m_sgemm_kernel;
 //   cl::Kernel & out_transform_bn_kernel =
 //       opencl_context.m_out_transform_bn_kernel;
 //   cl::Kernel & out_transform_bn_in_kernel =
 //       opencl_context.m_out_transform_bn_in_kernel;

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

    auto wgs = ceilMultiple(batch_size * tiles, wavefront_size);
    auto wgs_single = ceilMultiple(tiles, wavefront_size);

    auto m_ceil = int(ceilMultiple(ceilMultiple(outputs, mwg), vwm));
    auto n_ceil = int(ceilMultiple(ceilMultiple(batch_size * tiles, nwg), vwn));
    auto k_ceil = int(ceilMultiple(ceilMultiple(channels, kwg), vwm));

 //   cl::CommandQueue & queue = opencl_context.m_commandqueue;

    if (!skip_in_transform) {
		in_transform_host(bufferIn, bufferV, channels, k_ceil, n_ceil, batch_size);
    }


	//const auto offset_u = b * K * C;
	//const auto offset_v = b * C * P;
	//const auto offset_m = b * K * P;
	//cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	//			K, P, C,
	//			1.0f,
	//			&U[offset_u], K,
	//			&V[offset_v], P,
	//			0.0f,
	//			&M[offset_m], P);

	auto alpha = 1.0f;
	auto beta = 0.0f;

	//cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
    //                              cublasOperation_t transa,
    //                              cublasOperation_t transb,
    //                              int m, int n, int k,
    //                              const float           *alpha,
    //                              const float           *A, int lda,
    //                              long long int          strideA,
    //                              const float           *B, int ldb,
    //                              long long int          strideB,
    //                              const float           *beta,
    //                              float                 *C, int ldc,
    //                              long long int          strideC,
    //                              int batchCount)

	// Weights: [36][channels][outputs] = [36][k_ceil][m_ceil]
	// Input: [36][channels][tiles] = [36][k_ceil][n_ceil]
	// Output: [36][outputs][tiles] = [36][m_ceil][n_ceil]

	ReportCUBLASErrors(cublasSgemmStridedBatched(
					   opencl_context.m_cublas, // handle
					   CUBLAS_OP_N, // transa
					   CUBLAS_OP_T, // transb
					   m_ceil, // m
					   n_ceil, // n
					   k_ceil, // k
					   &alpha, // alpha
					   (float*)weights[0], // A
					   m_ceil, // lda
					   m_ceil * k_ceil, // strideA
					   (float*)bufferV, // B
					   n_ceil, // ldb
					   n_ceil * k_ceil, // strideB
					   &beta, // beta
					   (float*)bufferM, // C
					   m_ceil, // ldc
					   m_ceil * n_ceil, // strideC
					   WINOGRAD_TILE)); // batchCount

	//auto out_size = 20*m_ceil;
	//std::vector<float> out_temp(out_size);

	//ReportCUDAErrors(cudaMemcpy(&out_temp.data()[0], bufferM,
	//		  out_size * sizeof(float), cudaMemcpyDeviceToHost));

	//for (int i = 0; i < out_size/m_ceil; i++) {
	//	myprintf("%f ", out_temp[i*m_ceil]);
	//}
	//myprintf("\n\n");

    if (!skip_in_transform) {
		in_transform_host(bufferIn, bufferV, channels, k_ceil, n_ceil, batch_size);
    }

	if (fuse_in_transform) {
		// k_ceil of the next convolution
        auto k_ceil2 = int(ceilMultiple(ceilMultiple(outputs, kwg), vwm));

		fused_out_in_transform_host(bufferM, bufferOut, bufferV,
									outputs, m_ceil,
									n_ceil, k_ceil2,
									bufferResidual, bn_weights[0],
									bn_weights[1], batch_size);
	} else {
		out_transform_host(bufferM, bufferOut,
						   outputs, m_ceil,
						   n_ceil, batch_size,
						   bufferResidual, bn_weights[0], bn_weights[1]);
	}
}

template <typename net_t>
void OpenCL_Network<net_t>::convolve1(OpenCLContext & opencl_context,
                              int channels, int outputs,
                              void* bufferInput,
                              void* bufferOutput,
                              void* bufferMerge,
                              weight_slice_t weights,
                              int batch_size) {


	convolve1_host(channels, outputs,
				    bufferInput,
				    bufferOutput,
				    bufferMerge,
				    weights[0],
				    batch_size);


	//float alpha = 1.0f;
	//float beta = 0.0f;

	// Input: channels x (W x H) = k x m
	// Weights: channels x outputs = k x n
	// Output: outputs x (W x H) = n x m

	// Weights: [outputs][channels]
	// Input: [channels][wh]
	// Output: [outputs][wh]

	// A: k x m
	// B: n x k
	// C: n x m

	//ReportCUBLASErrors(cublasSgemmStridedBatched(
	//				   opencl_context.m_cublas, // handle
	//				   CUBLAS_OP_N, // transa
	//				   CUBLAS_OP_N, // transb
	//				   outputs, // m
	//				   NUM_INTERSECTIONS, // n
	//				   channels, // k
	//				   &alpha, // alpha
	//				   (float*)weights[0], // A
	//				   outputs, // lda
	//				   channels * outputs, // strideA
	//				   (float*)bufferInput, // B
	//				   channels, // ldb
	//				   channels * NUM_INTERSECTIONS, // strideB
	//				   &beta, // beta
	//				   (float*)bufferOutput, // C
	//				   outputs, // ldc
	//				   outputs * NUM_INTERSECTIONS, // strideC
	//				   batch_size));

//    // The size of the board is defined at compile time
//    constexpr int width = BOARD_SIZE;
//    constexpr int boardsize = NUM_INTERSECTIONS;
//    constexpr int rowTiles = BOARD_SIZE;
//
//    // Input channel grouping in multiples of 8
//    constexpr int channelGroup = 8;
//    constexpr int channelShift = 3;
//    constexpr int rowGroup = 1;
//    size_t outputGroup = std::min(outputs, 32);
//
//    auto m_convolve_kernel = &opencl_context.m_convolve1_kernel;
//
//#ifndef NDEBUG
//    // Total output size after reducing
//    size_t outSize = boardsize * outputs * sizeof(net_t);
//
//    // Produce channel * output planes and merge them at the end
//    size_t mergeSize = (channels >> channelShift) * outSize;
//    assert(mergeSize <= bufferMerge.getInfo<CL_MEM_SIZE>());
//#endif
//
//    // Copy the rows locally
//    size_t stripSize = width * sizeof(float);
//
//    int rowBuffer = std::min<int>(channelGroup, 7);
//    size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);
//
//    cl::CommandQueue & queue = opencl_context.m_commandqueue;
//
//    try {
//        m_convolve_kernel->setArg(0, bufferInput);
//        m_convolve_kernel->setArg(1, bufferMerge);
//        m_convolve_kernel->setArg(2, weights[0]);
//        m_convolve_kernel->setArg(3, cl::Local(stripSize * channelGroup * rowGroup));
//        m_convolve_kernel->setArg(4, cl::Local(rowSize));
//
//        queue.enqueueNDRangeKernel(
//            *m_convolve_kernel, cl::NullRange,
//            cl::NDRange(channels, outputs, batch_size * rowTiles),
//            cl::NDRange(channelGroup, outputGroup, rowGroup));
//    } catch (const cl::Error &e) {
//        std::cerr << "Error in convolve1: " << e.what() << ": "
//                  << e.err() << std::endl;
//        throw;
//    }
//
//    cl::Kernel & merge_kernel = opencl_context.m_merge_kernel;
//    assert(channels % (1 << channelShift) == 0);
//
//    try {
//        merge_kernel.setArg(0, bufferMerge);
//        merge_kernel.setArg(1, bufferOutput);
//        merge_kernel.setArg(2, channels >> channelShift);
//
//        queue.enqueueNDRangeKernel(
//            merge_kernel, cl::NullRange,
//            cl::NDRange(outputs, boardsize, batch_size),
//            cl::NDRange(std::min(8, outputs), BOARD_SIZE, 1));
//    } catch (const cl::Error &e) {
//        std::cerr << "Error in merge: " << e.what() << ": "
//                  << e.err() << std::endl;
//        throw;
//    }
}

template<class T>
static std::string opencl_dev_type_to_string(T type) {
    return "GPU";
}

static std::string trim(std::string trim_me) {
    boost::algorithm::trim(trim_me);
    return trim_me;
}

template <typename net_t>
void OpenCL<net_t>::process_tuners(std::string tuners) {
    auto tile_size = 32;
    m_sgemm_tuners.mwg = tile_size;
    m_sgemm_tuners.nwg = tile_size;
    m_sgemm_tuners.kwg = tile_size;
    m_sgemm_tuners.mdimc = 1;
    m_sgemm_tuners.ndimc = 1;
    m_sgemm_tuners.vwm = 1;
    m_sgemm_tuners.vwn = 1;
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
OpenCL<net_t>::OpenCL(int gpu, bool silent) {

    auto best_bandwidth = 0.0f;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    cudaGetDeviceCount(&nDevices);

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    auto id = 0;

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto bandwidth = 2.0f*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak memory bandwidth (GB/s): %.1f\n\n",
                   bandwidth);
        }

        bool preferred = (gpu == id);

        if ( (bandwidth > best_bandwidth) || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
        id++;
    }

    if (!found_device) {
        throw std::runtime_error("No suitable CUDA device found.");
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    cudaSetDevice(best_device_id);

    m_fp16_compute = false;
}

template <typename net_t>
void OpenCL<net_t>::initialize(const int channels) {
        /* For compatibility with OpenCL implementation */
    (void)channels;
    process_tuners("");

	// Hard coded for now
    m_wavefront_size = 32;

    // Build program for these specific devices
    //try {
    //    std::string args = m_cl_args;
    //    args += sgemm_tuners;
    //    m_program.build(args.c_str());
    //} catch (const cl::Error&) {
    //    myprintf("Error building kernels: %s\n",
    //             m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device).c_str());
    //    throw std::runtime_error("Error building OpenCL kernels.");
    //}

    //OpenCLContext tdata;
    //ensure_context_initialized(tdata);

    //process_tuners(sgemm_tuners);

    //m_wavefront_size =
    //    tdata.m_sgemm_kernel.getWorkGroupInfo<
    //        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(m_device);
    //myprintf("Wavefront/Warp size: %d\n", m_wavefront_size);

    //m_max_workgroup_size = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    //m_max_workgroup_dims = m_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    //myprintf("Max workgroup size: %d\n", m_max_workgroup_size);
    //myprintf("Max workgroup dimensions: ");
    //for (auto d : m_max_workgroup_dims) {
    //    myprintf("%d ", d);
    //}
    //myprintf("\n");

    m_init_ok = true;
}

template <typename net_t>
bool OpenCL<net_t>::has_fp16_compute() {
    return m_fp16_compute;
}

template <typename net_t>
std::string OpenCL<net_t>::get_device_name() {
    std::stringstream ss;

    ss << "CUDA: ";
    //ss << m_device.getInfo<CL_DEVICE_VENDOR>() << " ";
    //ss << m_device.getInfo<CL_DEVICE_NAME>() << " @ ";
    //ss << m_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

    return ss.str();
}

template class OpenCL<float>;
template class OpenCL_Network<float>;
#ifdef USE_HALF
template class OpenCL<half_float::half>;
template class OpenCL_Network<half_float::half>;
#endif

#endif
