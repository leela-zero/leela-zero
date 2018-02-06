
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

// Merges the results in Cpm with the global array in Cgm.
INLINE_FUNC void StoreResults(__global realM* cgm, realM cpm[NWI*MWI/VWM], const int kSizeM) {
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

      cgm[index] = cpm[_ni * (MWI/VWM) + _mi];

    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
