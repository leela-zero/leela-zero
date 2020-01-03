
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xgemv kernel (generic version) for matrix-vector multiplication.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// 1: For the full version of the kernel
#ifndef WGS1
  #define WGS1 32     // The local work-group size
#endif
#ifndef WPT1
  #define WPT1 1      // The amount of work-per-thread
#endif
#ifndef UNROLL1
  #define UNROLL1 32  // Unroll factor (must be a divider of WGS1)
#endif

// 2 and 3: For the fast versions, see 'xgemv_fast.opencl'

// =================================================================================================

// Defines how to load the input matrix in the non-vectorized case
INLINE_FUNC real LoadMatrixA(const __global real* restrict agm, const int x, const int y,
                             const int a_ld, const int a_offset) {

#ifdef FP16_STORAGE
  return vloada_half(a_ld*y + x + a_offset, (const __global half*)agm);
#else
  return agm[a_ld*y + x + a_offset];
#endif
}

INLINE_FUNC real LoadValue(const __global real* restrict x, const int offset) {

#ifdef FP16_STORAGE
  return vloada_half(offset, (const __global half*)x);
#else
  return x[offset];
#endif
}

INLINE_FUNC void StoreValue(__global real* restrict y, const int offset, const real value) {

#ifdef FP16_STORAGE
  vstorea_half(value, offset, (__global half*)y);
#else
  y[offset] = value;
#endif
}

// =================================================================================================

// Full version of the kernel
__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xgemv(const int m, const int n,
                    const __global real* restrict agm, const int a_offset, const int a_ld,
                    const __global real* restrict xgm, const int x_offset,
                    __global real* ygm, const int y_offset,
                    __global real* bias, const int relu) {

  const int batch = get_global_id(1);

  // Local memory for the vector X
  __local real xlm[WGS1];

  // Initializes the accumulation register
  #pragma promote_to_registers
  real acc1[WPT1];
  #pragma unroll
  for (int _w = 0; _w < WPT1; _w += 1) {
    SetToZero(acc1[_w]);
  }

  // Divides the work in a main and tail section
  const int n_tail = n % WGS1;
  const int n_floor = n - n_tail;

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<n_floor; kwg+=WGS1) {

    // Loads the vector X into local memory
    const int lid = get_local_id(0);
    xlm[lid] = LoadValue(xgm, (kwg + lid) + x_offset + batch * n);

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loops over the work per thread, and checks whether in bounds
    #pragma unroll
    for (int _w = 0; _w < WPT1; _w += 1) {
      const int gid = _w*get_global_size(0) + get_global_id(0);
      if (gid < m) {

        // The multiply-add function for the main part (divisable by WGS1)
        for (int kloop=0; kloop<WGS1; kloop+=UNROLL1) {
          #pragma unroll
          for (int _kunroll = 0; _kunroll < UNROLL1; _kunroll += 1) {
            const int k = kwg + kloop + _kunroll;
            real value = LoadMatrixA(agm, k, gid, a_ld, a_offset);
            MultiplyAdd(acc1[_w], xlm[kloop + _kunroll], value);
          }
        }
      }
    }

    // Synchronizes all threads in a workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Loops over the work per thread, and checks whether in bounds
  #pragma unroll
  for (int _w = 0; _w < WPT1; _w += 1) {
    const int gid = _w*get_global_size(0) + get_global_id(0);
    if (gid < m) {

      // The multiply-add function for the remainder part (not divisable by WGS1)
      for (int k=n_floor; k<n; ++k) {
        real value = LoadMatrixA(agm, k, gid, a_ld, a_offset);
        const real x_k = LoadValue(xgm, k + x_offset + batch * n);
        MultiplyAdd(acc1[_w], x_k, value);

      }

      // Stores the final result
      real out = acc1[_w] + LoadValue(bias, gid + y_offset);
      if (relu) {
        out = out > 0.0f ? out : 0.0f;
      }
      StoreValue(ygm, gid + y_offset + batch * m, out);
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
