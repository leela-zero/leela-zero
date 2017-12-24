
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

#define PRECISION 32
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

  // Enable support for double-precision
  #if PRECISION == 64 || PRECISION == 6464
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
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

// Double-precision 
#elif PRECISION == 64
  typedef double real;
  typedef double2 real2;
  typedef double4 real4;
  typedef double8 real8;
  typedef double16 real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37

// Complex single-precision
#elif PRECISION == 3232
  typedef float2 real;
  typedef struct cfloat2 {real x; real y;} real2;
  typedef struct cfloat4 {real x; real y; real z; real w;} real4;
  typedef struct cfloat8 {real s0; real s1; real s2; real s3;
                          real s4; real s5; real s6; real s7;} real8;
  typedef struct cfloat16 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;
                           real s8; real s9; real sA; real sB;
                           real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Complex double-precision
#elif PRECISION == 6464
  typedef double2 real;
  typedef struct cdouble2 {real x; real y;} real2;
  typedef struct cdouble4 {real x; real y; real z; real w;} real4;
  typedef struct cdouble8 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;} real8;
  typedef struct cdouble16 {real s0; real s1; real s2; real s3;
                            real s4; real s5; real s6; real s7;
                            real s8; real s9; real sA; real sB;
                            real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37
#endif

// Single-element version of a complex number
#if PRECISION == 3232
  typedef float singlereal;
#elif PRECISION == 6464
  typedef double singlereal;
#else
  typedef real singlereal;
#endif

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
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
  #define IsZero(a) (a == ZERO)
#endif

// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)
#else
  #define AbsoluteValue(value) value = fabs(value)
#endif

// Negation (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define Negate(value) value.x = -(value.x); value.y = -(value.y)
#else
  #define Negate(value) value = -(value)
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y
#else
  #define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a,b) a.x*b.x - a.y*b.y
  #define MulImag(a,b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c,a,b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #if USE_CL_MAD == 1
    #define MultiplyAdd(c,a,b) c = mad(a, b, c)
  #else
    #define MultiplyAdd(c,a,b) c += a * b
  #endif
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)
#else
  #define MultiplySubtract(c,a,b) c -= a * b
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
  #define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom
#else
  #define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value) 
#endif

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
