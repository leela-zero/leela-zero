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

#include "Utils.h"
#include <sstream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

using namespace Utils;

#define OUT_KWG 8
#define OUT_BWG 8
#define OUTIN_KWG 2

#ifdef USE_OPENCL

int DivUp(int a, int b) { return (a + b - 1) / b; }

void CudaError(cudaError_t status, const std::string& file, const std::string& line) {
  if (status != cudaSuccess) {
    std::ostringstream stringStream;
    stringStream << "CUDA error: " << std::string(cudaGetErrorString(status))
                 << " (" << file
                 << ": " << line << ")";
    throw std::runtime_error(stringStream.str());
  }
}


std::string CublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cuBlas error";
}


void CublasError(cublasStatus_t status, const std::string& file, const std::string& line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream stringStream;
    stringStream << "Cublas error: " << CublasGetErrorString(status)
                 << " (" << file
                 << ": " << line << ")";
    throw std::runtime_error(stringStream.str());
  }
}

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define ReportCUBLASErrors(status) CublasError((status), __FILE__, TOSTRING(__LINE__))
#define ReportCUDAErrors(status) CudaError((status), __FILE__, TOSTRING(__LINE__))

// Winograd defines. TODO: include from Network.h

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
#define WINOGRAD_M 4
#define WINOGRAD_ALPHA (WINOGRAD_M + 3 - 1)
#define WTILES (BOARD_SIZE / WINOGRAD_M + (BOARD_SIZE % WINOGRAD_M != 0))
#define WINOGRAD_TILE (WINOGRAD_ALPHA * WINOGRAD_ALPHA)
#define WINOGRAD_P (WTILES * WTILES)

#include "kernels/opencl_to_cuda.h"
#include "kernels/common.opencl"
#include "kernels/convolve1.opencl"
#include "kernels/convolve3.opencl"

void in_transform_host(void* in, void* V,
                  const int C, const int Cpad,
                  const int Ppad, const int batch_size) {
    auto tiles = WINOGRAD_P;

    auto wgs = ceilMultiple(batch_size * tiles, 32);

    dim3 threads( ceilMultiple(tiles, 32), 1 );
    dim3 grid( DivUp(wgs, threads.x), DivUp(C, threads.y) );

    in_transform<<<grid, threads>>>((float*)in, (float*)V, C, Cpad, Ppad, batch_size);

    ReportCUDAErrors(cudaGetLastError());
}

void out_transform_host(void* M, void* Y,
                  const int K, const int Kpad,
                  const int Ppad, const int batch_size, void** residual,
				  void* means, void* stddivs) {
    auto tiles = WINOGRAD_P;

    auto wgs = ceilMultiple(batch_size * tiles, 32);

    dim3 threads( OUT_KWG, OUT_BWG );
    dim3 grid( DivUp(K, threads.x), DivUp(wgs, threads.y) );

	void *residual_device = nullptr;
	if (residual != nullptr) {
		residual_device = *residual;
	}

    out_transform_fused_bn<<<grid, threads>>>((float*)M, (float*)Y, K, Kpad, Ppad, batch_size, (float*)residual_device, (float*)means, (float*)stddivs);

    ReportCUDAErrors(cudaGetLastError());
}

void fused_out_in_transform_host(void* M, void* Y, void* V,
                                 const int K, const int Kpad,
								 const int Ppad, const int Cpad,
                                 void** residual, void* means, void* stddivs,
                                 const int batch_size) {
    auto tiles = WINOGRAD_P;

    auto wgs_single = ceilMultiple(tiles, 32);

    dim3 threads( OUTIN_KWG, wgs_single, 1 );
    dim3 grid( DivUp(K, threads.x), 1, batch_size );

	void *residual_device = nullptr;
	if (residual != nullptr) {
		residual_device = *residual;
	}

    out_transform_fused_bn_in<<<grid, threads>>>((float*)M, (float*)Y, (float*)V, K, Kpad, Ppad, Cpad, (float*)residual_device, (float*)means, (float*)stddivs);

    ReportCUDAErrors(cudaGetLastError());
}

void convolve1_host(int channels, int outputs,
				    void* bufferInput,
				    void* bufferOutput,
				    void* bufferMerge,
				    void* weights,
				    int batch_size) {
    // The size of the board is defined at compile time
    const int width = BOARD_SIZE;
    const int boardsize = NUM_INTERSECTIONS;
    const int rowTiles = BOARD_SIZE;

    // Input channel grouping in multiples of 8
    const int channelGroup = 8;
    const int channelShift = 3;
    const int rowGroup = 1;
    size_t outputGroup = std::min(outputs, 32);

    // Copy the rows locally
    size_t stripSize = width * sizeof(float);

    int rowBuffer = std::min<int>(channelGroup, 7);
    size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);

    dim3 threads( channelGroup, outputGroup, rowGroup );
    dim3 grid( DivUp(channels, threads.x), DivUp(outputs, threads.y), DivUp(batch_size * rowTiles, threads.z));

	size_t shared_mem = stripSize * channelGroup * rowGroup + rowSize;

	convolve1<<<grid, threads, shared_mem>>>((float*)bufferInput, (float*)bufferMerge,
			(float*)weights);

    dim3 threads_merge( std::min(8, outputs), BOARD_SIZE, 1);
    dim3 grid_merge( DivUp(outputs, threads_merge.x), DivUp(NUM_INTERSECTIONS, threads_merge.y), DivUp(batch_size, threads_merge.z));

	merge<<<grid_merge, threads_merge>>>((float*)bufferMerge,
			(float*)bufferOutput, channels >> channelShift);

    ReportCUDAErrors(cudaGetLastError());
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

#endif
