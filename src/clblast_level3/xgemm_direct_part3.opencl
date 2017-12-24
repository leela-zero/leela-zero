
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
INLINE_FUNC void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                             const real_arg arg_alpha,
                             const real_arg arg_beta,
                             const __global realMD* restrict agm, const int a_offset, const int a_ld,
                             const __global realND* restrict bgm, const int b_offset, const int b_ld,
                             __global real* cgm, const int c_offset, const int c_ld,
                             LOCAL_PTR real* alm, LOCAL_PTR real* blm,
                             const int a_transpose, const int b_transpose, const int c_transpose,
                             const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  real apd[MWID];
  #pragma promote_to_registers
  real bpd[NWID];
  #pragma promote_to_registers
  real cpd[NWID * MWID];

  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWID; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      SetToZero(cpd[_ni * MWID + _mi]);
    }
  }

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
        GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0 && b_offset % VWND == 0) {
        GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
          }

          // Performs the accumulation (Cpmd += Apmd * Bpmd)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
      }

      // Performs the accumulation (Cpmd += Apmd * Bpmd)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    StoreResultsDirect(cgm, cpd, idm, idn, alpha, beta, c_ld, c_offset, c_transpose);
  }

  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
      GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
          }

          // Performs the accumulation (C += A * B)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateCheckedA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateCheckedB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
      }

      // Performs the accumulation (C += A * B)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    StoreResultsChecked(cgm, cpd, idm, idn, kSizeM, kSizeN, alpha, beta, c_ld, c_offset, c_transpose);
  }
}

// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
