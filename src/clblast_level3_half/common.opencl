
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common defines and type-defs for the CLBlast OpenCL kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

#define ROUTINE_GEMMBATCHED

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif

// =================================================================================================
#ifndef CUDA
  // Enable support for double-precision
  #if PRECISION == 16
    #pragma OPENCL EXTENSION cl_khr_fp16: enable
  #endif
#endif

// Half-precision
#if PRECISION == 16
  typedef half real;
  typedef half2 real2;
  typedef half4 real4;
  typedef half8 real8;
  typedef half16 real16;
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f
#endif

// Single-element version of a complex number
  typedef real singlereal;

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
  typedef float real_arg;
  #define GetRealArg(x) (half)x
#else
  typedef real real_arg;
  #define GetRealArg(x) x
#endif

// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
  #define LOCAL_PTR __local
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

// Sets a variable to zero
#define SetToZero(a) a = ZERO

// Sets a variable to zero (only the imaginary part)
#define ImagToZero(a)

// Sets a variable to one
#define SetToOne(a) a = ONE

// Determines whether a variable is zero
#define IsZero(a) (a == ZERO)

// The absolute value (component-wise)
#define AbsoluteValue(value) value = fabs(value)

// Negation (component-wise)
#define Negate(value) value = -(value)

// Adds two complex variables
#define Add(c,a,b) c = a + b

// Subtracts two complex variables
#define Subtract(c,a,b) c = a - b

// The scalar multiply function
#define Multiply(c,a,b) c = a * b

// The scalar multiply-add function
#if USE_CL_MAD == 1
  #define MultiplyAdd(c,a,b) c = mad(a, b, c)
#else
  #define MultiplyAdd(c,a,b) c += a * b
#endif

// The scalar multiply-subtract function
#define MultiplySubtract(c,a,b) c -= a * b

// The scalar division function: full division
#define DivideFull(c,a,b) c = a / b

// The scalar AXPBY function
#define AXPBY(e,a,b,c,d) e = a*b + c*d

// The complex conjugate operation for complex transforms
#define COMPLEX_CONJUGATE(value)

// =================================================================================================

// Force inlining functions or not: some compilers don't support the inline keyword
#ifdef USE_INLINE_KEYWORD
  #define INLINE_FUNC inline
#else
  #define INLINE_FUNC
#endif

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
  #define USE_STAGGERED_INDICES 0
#endif

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1
  INLINE_FUNC int GetGroupIDFlat() {
    return get_group_id(0) + get_num_groups(0) * get_group_id(1);
  }
  INLINE_FUNC int GetGroupID1() {
    return (GetGroupIDFlat()) % get_num_groups(1);
  }
  INLINE_FUNC int GetGroupID0() {
    return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
  }
#else
  INLINE_FUNC int GetGroupID1() { return get_group_id(1); }
  INLINE_FUNC int GetGroupID0() { return get_group_id(0); }
#endif

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
