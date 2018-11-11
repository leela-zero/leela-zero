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

#ifndef CUDA_KERNELS_H_INCLUDED
#define CUDA_KERNELS_H_INCLUDED

#include <cuda.h>
#include <cublas_v2.h>

void CudaError(cudaError_t status, const std::string& file, const std::string& line);
void CublasError(cublasStatus_t status, const std::string& file, const std::string& line);

std::string CublasGetErrorString(cublasStatus_t status);


void in_transform_host(void* in, void* V,
                  const int C, const int Cpad,
                  const int Ppad, const int batch_size);

void out_transform_host(void* M, void* Y,
                  const int K, const int Kpad,
                  const int Ppad, const int batch_size, void** residual,
				  void* means, void* stddivs);

void fused_out_in_transform_host(void* M, void* Y, void* V,
                                 const int K, const int Kpad,
								 const int Ppad, const int Cpad,
                                 void** residual, void* means, void* stddivs,
                                 const int batch_size);

void convolve1_host(int channels, int outputs,
				    void* bufferInput,
				    void* bufferOutput,
				    void* bufferMerge,
				    void* weights,
				    int batch_size);

#endif
