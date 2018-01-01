
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// The vectorised multiply-add function
INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWM == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWM == 2
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
    #elif VWM == 4
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
      MultiplyAdd(cvec.z , avec.z,  bval);
      MultiplyAdd(cvec.w , avec.w,  bval);
    #elif VWM == 8
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
    #elif VWM == 16
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
      MultiplyAdd(cvec.s8, avec.s8, bval);
      MultiplyAdd(cvec.s9, avec.s9, bval);
      MultiplyAdd(cvec.sA, avec.sA, bval);
      MultiplyAdd(cvec.sB, avec.sB, bval);
      MultiplyAdd(cvec.sC, avec.sC, bval);
      MultiplyAdd(cvec.sD, avec.sD, bval);
      MultiplyAdd(cvec.sE, avec.sE, bval);
      MultiplyAdd(cvec.sF, avec.sF, bval);
    #endif
  #endif
  return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResults(__global realM* cgm, realM cpm[NWI*MWI/VWM], const int kSizeM,
                              const real alpha, const real beta) {
  #pragma unroll
  for (int _ni = 0; _ni < NWI; _ni += 1) {
    #pragma unroll
    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
      #if STRM == 0
        int mg = _mi + get_local_id(0)*(MWI/VWM);
      #elif STRM == 1
        int mg = get_local_id(0) + _mi*MDIMC;
      #endif
      #if STRN == 0
        int ng = _ni + get_local_id(1)*NWI;
      #elif STRN == 1
        int ng = _ni%VWN + get_local_id(1)*VWN + (_ni/VWN)*VWN*NDIMC;
      #endif
      int idm = mg + GetGroupID0() * (MWG/VWM);
      int idn = ng + GetGroupID1() * NWG;
      int index = idn*(kSizeM/VWM) + idm;

      realM result;
      realM xval = cpm[_ni * (MWI/VWM) + _mi];

      // The final multiplication with alpha (in case beta == 0)
      if (IsZero(beta)) {
        #if VWM == 1
          Multiply(result, alpha, xval);
        #elif VWM == 2
          Multiply(result.x, alpha, xval.x);
          Multiply(result.y, alpha, xval.y);
        #elif VWM == 4
          Multiply(result.x, alpha, xval.x);
          Multiply(result.y, alpha, xval.y);
          Multiply(result.z, alpha, xval.z);
          Multiply(result.w, alpha, xval.w);
        #elif VWM == 8
          Multiply(result.s0, alpha, xval.s0);
          Multiply(result.s1, alpha, xval.s1);
          Multiply(result.s2, alpha, xval.s2);
          Multiply(result.s3, alpha, xval.s3);
          Multiply(result.s4, alpha, xval.s4);
          Multiply(result.s5, alpha, xval.s5);
          Multiply(result.s6, alpha, xval.s6);
          Multiply(result.s7, alpha, xval.s7);
        #elif VWM == 16
          Multiply(result.s0, alpha, xval.s0);
          Multiply(result.s1, alpha, xval.s1);
          Multiply(result.s2, alpha, xval.s2);
          Multiply(result.s3, alpha, xval.s3);
          Multiply(result.s4, alpha, xval.s4);
          Multiply(result.s5, alpha, xval.s5);
          Multiply(result.s6, alpha, xval.s6);
          Multiply(result.s7, alpha, xval.s7);
          Multiply(result.s8, alpha, xval.s8);
          Multiply(result.s9, alpha, xval.s9);
          Multiply(result.sA, alpha, xval.sA);
          Multiply(result.sB, alpha, xval.sB);
          Multiply(result.sC, alpha, xval.sC);
          Multiply(result.sD, alpha, xval.sD);
          Multiply(result.sE, alpha, xval.sE);
          Multiply(result.sF, alpha, xval.sF);
        #endif
      }

      // The final multiplication with alpha and the addition with beta*C
      else {
        realM yval = cgm[index];
        #if VWM == 1
          AXPBY(result, alpha, xval, beta, yval);
        #elif VWM == 2
          AXPBY(result.x, alpha, xval.x, beta, yval.x);
          AXPBY(result.y, alpha, xval.y, beta, yval.y);
        #elif VWM == 4
          AXPBY(result.x, alpha, xval.x, beta, yval.x);
          AXPBY(result.y, alpha, xval.y, beta, yval.y);
          AXPBY(result.z, alpha, xval.z, beta, yval.z);
          AXPBY(result.w, alpha, xval.w, beta, yval.w);
        #elif VWM == 8
          AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
          AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
          AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
          AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
          AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
          AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
          AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
          AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
        #elif VWM == 16
          AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
          AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
          AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
          AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
          AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
          AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
          AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
          AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
          AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);
          AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);
          AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);
          AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);
          AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);
          AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);
          AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);
          AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);
        #endif
      }
      cgm[index] = result;
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
