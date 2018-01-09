
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for level 3 BLAS kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// For the 'fast' copy kernel
#ifndef COPY_DIMX
  #define COPY_DIMX 8      // Local workgroup size in the first dimension (x)
#endif
#ifndef COPY_DIMY
  #define COPY_DIMY 8      // Local workgroup size in the second dimension (y)
#endif
#ifndef COPY_WPT
  #define COPY_WPT 1       // Work per thread in the first dimension (x)
#endif
#ifndef COPY_VW
  #define COPY_VW 1        // Vector width in the second dimension (y)
#endif

// For the padding/copy kernels and the conversion kernels
#ifndef PAD_DIMX
  #define PAD_DIMX 8      // Local workgroup size in the first dimension (x)
#endif
#ifndef PAD_DIMY
  #define PAD_DIMY 8      // Local workgroup size in the second dimension (y)
#endif
#ifndef PAD_WPTX
  #define PAD_WPTX 1      // Work per thread in the first dimension (x)
#endif
#ifndef PAD_WPTY
  #define PAD_WPTY 1      // Work per thread in the second dimension (y)
#endif

// For the 'fast' transpose kernel
#ifndef TRA_DIM
  #define TRA_DIM 8       // Number of local threads in the two dimensions (x,y)
#endif
#ifndef TRA_WPT
  #define TRA_WPT 1       // Work per thread in one dimension and vector-width in the other
#endif
#ifndef TRA_PAD
  #define TRA_PAD 0       // Padding of the local memory to avoid bank-conflicts
#endif
#ifndef TRA_SHUFFLE
  #define TRA_SHUFFLE 0   // Shuffling of the global indices to avoid global memory bank-conflicts
#endif

// For the padding/transpose kernels
#ifndef PADTRA_TILE
  #define PADTRA_TILE 8   // Number of local threads in the two dimensions (x,y)
#endif
#ifndef PADTRA_WPT
  #define PADTRA_WPT 1    // Amount of work per thread
#endif
#ifndef PADTRA_PAD
  #define PADTRA_PAD 0    // Padding of the local memory to avoid bank-conflicts
#endif

// =================================================================================================
#if defined(ROUTINE_INVERT) || defined(ROUTINE_TRSM)

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void FillMatrix(const int m, const int n, const int ld, const int offset,
                __global real* restrict dest, const real_arg arg_value) {
  const real value = GetRealArg(arg_value);
  const int id_one = get_global_id(0);
  const int id_two = get_global_id(1);
  if (id_one < m && id_two < n) {
    dest[id_two*ld + id_one + offset] = value;
  }
}

#endif

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
