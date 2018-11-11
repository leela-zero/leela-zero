
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an (incomplete) header to interpret OpenCL kernels as CUDA kernels.
//
// =================================================================================================

// =================================================================================================

// CLBlast specific additions
#define CUDA 1
#define LOCAL_PTR  // pointers to local memory don't have to be annotated in CUDA

// Replaces the OpenCL get_xxx_ID with CUDA equivalents
__device__ int get_local_id(const int x) {
  if (x == 0) { return threadIdx.x; }
  if (x == 1) { return threadIdx.y; }
  return threadIdx.z;
}
__device__ int get_group_id(const int x) {
  if (x == 0) { return blockIdx.x; }
  if (x == 1) { return blockIdx.y; }
  return blockIdx.z;
}
__device__ int get_local_size(const int x) {
  if (x == 0) { return blockDim.x; }
  if (x == 1) { return blockDim.y; }
  return blockDim.z;
}
__device__ int get_num_groups(const int x) {
  if (x == 0) { return gridDim.x; }
  if (x == 1) { return gridDim.y; }
  return gridDim.z;
}
__device__ int get_global_size(const int x) {
  if (x == 0) { return gridDim.x * blockDim.x; }
  if (x == 1) { return gridDim.y * blockDim.y; }
  return gridDim.z * blockDim.z;
}
__device__ int get_global_id(const int x) {
  if (x == 0) { return blockIdx.x*blockDim.x + threadIdx.x; }
  if (x == 1) { return blockIdx.y*blockDim.y + threadIdx.y; }
  return blockIdx.z*blockDim.z + threadIdx.z;
}

// Adds the data-types which are not available natively under CUDA
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7; } float8;
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7;
                 float s8; float s9; float s10; float s11;
                 float s12; float s13; float s14; float s15; } float16;
typedef struct { double s0; double s1; double s2; double s3;
                 double s4; double s5; double s6; double s7; } double8;
typedef struct { double s0; double s1; double s2; double s3;
                 double s4; double s5; double s6; double s7;
                 double s8; double s9; double s10; double s11;
                 double s12; double s13; double s14; double s15; } double16;

// Replaces the OpenCL keywords with CUDA equivalent
#define __kernel __placeholder__
#define __global
#define __placeholder__ extern "C" __global__
#define __local __shared__
#define restrict __restrict__
#define __constant const
#define inline __device__ // assumes all device functions are annotated with inline in OpenCL

// Kernel attributes (don't replace currently)
#define reqd_work_group_size(x, y, z)

// Replaces OpenCL synchronisation with CUDA synchronisation
#define barrier(x) __syncthreads()

// =================================================================================================

// =================================================================================================

