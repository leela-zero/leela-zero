// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define USE_TC

#ifndef SA
#define SA 1
#endif

#ifndef SB
#define SB 1
#endif

#ifndef VWM
#define VWM 4
#endif

#ifndef VWN
#define VWN 2
#endif

#if VWM == 1
#define vstoreM vstore
#define vloadM vload
#elif VWM == 2
#define vstoreM vstore2
#define vloadM vload2
#elif VWM == 4
#define vstoreM vstore4
#define vloadM vload4
#elif VWM == 8
#define vstoreM vstore8
#define vloadM vload8
#elif VWM == 16
#define vstoreM vstore16
#define vloadM vload16
#endif

#if VWN == 1
#define vstoreN vstore
#define vloadN vload
#elif VWN == 2
#define vstoreN vstore2
#define vloadN vload2
#elif VWN == 4
#define vstoreN vstore4
#define vloadN vload4
#elif VWN == 8
#define vstoreN vstore8
#define vloadN vload8
#elif VWN == 16
#define vstoreN vstore16
#define vloadN vload16
#endif

void GlobalToLocalA(int tid, int stride, __local short * alm, __global short * agm)
{
    const int copy_size = KWG * MWG;
    const int dest_stride = MWG;
    const int num_threads = MDIMC * NDIMC * 32 / 256;

    for(int i=tid * VWM; i < copy_size; i += num_threads * VWM) {
        int x = i % dest_stride;
        int y = i / dest_stride;

        vstoreM( vloadM((y * stride + x) / VWM, agm), i / VWM, alm);
    } 
}


void GlobalToLocalB(int tid, int stride, __local short * blm, __global short * bgm)
{
    const int copy_size = KWG * NWG;
    const int dest_stride = NWG;
    const int num_threads = MDIMC * NDIMC * 32 / 256;
    for(int i=tid; i < copy_size; i += num_threads) {
        int x = i % dest_stride;
        int y = i / dest_stride;
        vstoreN( vloadN((y * stride + x) / VWN, bgm), i / VWN, blm);
    } 
}


void HgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                  #if SA == 1
                    __local short* alm,
                  #endif
                  #if SB == 1
                    __local short* blm,
                  #endif
                  const __global half* restrict agm,
                  const __global half* restrict bgm,
                  __global half* restrict cgm)
{
    int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    // the base location of the 16x16 tile number this thread is responsible of
    int tile_m = get_global_id(0) / 32 * MWG / MDIMC;
    int tile_n = get_global_id(1) * NWG / NDIMC;

    // the base pointers of agm, bgm and cgm
    const __global half * agm_ = agm + 16 * tile_m;
    const __global half * bgm_ = bgm + 16 * tile_n;
    __global half * cgm_ = cgm + kSizeM * 16 * tile_n + 16 * tile_m;

    // the (m,n) position within the warp
    int offset_number = laneid;
    int offset_m = offset_number % 8;
    int offset_n = offset_number / 8;
    
    if(laneid != get_global_id(0) % 32) {
        return;
    }

    int k, m, n, mb, nb, kb, kwg;
#ifdef USE_TC
    int zero_pair;
    asm("{\n"
        ".reg .b16 xh;\n"
        ".reg .b32 x;\n"
        "mov.f32 x, 0.0;\n"
        "cvt.rz.f16.f32 xh, x;\n"
        "mov.b32 %0, {xh,xh};\n"
        "}": "=r"(zero_pair)
    );

#pragma promote_to_registers
    int c0[MWG/MDIMC][NWG/NDIMC];
#pragma promote_to_registers
    int c1[MWG/MDIMC][NWG/NDIMC];
#pragma promote_to_registers
    int c2[MWG/MDIMC][NWG/NDIMC];
#pragma promote_to_registers
    int c3[MWG/MDIMC][NWG/NDIMC];
    #pragma unroll
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        #pragma unroll
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            c0[mb][nb] = zero_pair;
            c1[mb][nb] = zero_pair;
            c2[mb][nb] = zero_pair;
            c3[mb][nb] = zero_pair;
        }
    }
#else
    float acc[MWG/MDIMC][NWG/NDIMC][2][4];
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            for(m=0; m<2; m++) {
                for(int n=0; n<4; n++) {
                    acc[mb][nb][m][n] = 0.0f;
                }
            }
        }
    }
#endif
    for(kwg = 0; kwg < kSizeK; kwg += KWG) {
#if SA == 1
        GlobalToLocalA(get_local_id(0) +  get_local_id(1) * 32 * MDIMC / 16, kSizeM,
            alm, 
            (__global short *)(agm + get_group_id(0) * MWG + kwg * kSizeM)
        ); 
#endif

#if SB == 1
        GlobalToLocalB(get_local_id(0) +  get_local_id(1) * 32 * MDIMC, kSizeN,
            blm, 
            (__global short *)(bgm + get_group_id(1) * NWG + kwg * kSizeN)
        ); 

#endif

#if SA == 1 || SB == 1
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

#pragma unroll
        for(kb = 0; kb < KWG; kb += 16) {
#pragma unroll
            for(mb = 0; mb < MWG / MDIMC; mb += 1) {
#pragma unroll
                for(nb = 0; nb < NWG / NDIMC; nb += 1) {
#if SA == 1
                    const int block_loc_m = (get_local_id(0)/32) % (MDIMC/16);
                    const int agm_stride = MWG;
                    const __local half * b_agm_ = (const __local half *)(alm + (mb + block_loc_m * (MWG/MDIMC)) * 16);
                    const __local half * bb_agm_ = b_agm_ + agm_stride * kb;
#else
                    const int agm_stride = kSizeM;
                    const __global half * b_agm_ = agm_ + mb * 16;
                    const __global half * bb_agm_ = b_agm_ + kSizeM * (kb + kwg);
#endif

#if SB == 1
                    const int block_loc_n = (get_local_id(1)) % (NDIMC/16);
                    const int bgm_stride = NWG;
                    const __local half * b_bgm_ = (const __local half *)(blm + (nb + block_loc_n * (NWG/NDIMC)) * 16);
                    const __local half * bb_bgm_ = b_bgm_ + bgm_stride * kb;
#else
                    const int bgm_stride = kSizeN;
                    const __global half * b_bgm_ = bgm_ + nb * 16;
                    const __global half * bb_bgm_ = b_bgm_ + kSizeN * (kb + kwg);
#endif
#ifdef USE_TC
                    int d0_, d1_, d2_, d3_;
                    int c0_ = c0[mb][nb];
                    int c1_ = c1[mb][nb];
                    int c2_ = c2[mb][nb];
                    int c3_ = c3[mb][nb];
                    asm("{\n"
                        ".reg .b32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                        ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n"
#if SA == 1
                        "wmma.load.a.sync.aligned.m16n16k16.shared.col.f16 {a0,a1,a2,a3,a4,a5,a6,a7}, [%4], %6;\n"
#else
                        "wmma.load.a.sync.aligned.m16n16k16.global.col.f16 {a0,a1,a2,a3,a4,a5,a6,a7}, [%4], %6;\n"
#endif
#if SB == 1
                        "wmma.load.b.sync.aligned.m16n16k16.shared.row.f16 {b0,b1,b2,b3,b4,b5,b6,b7}, [%5], %7;\n"
#else
                        "wmma.load.b.sync.aligned.m16n16k16.global.row.f16 {b0,b1,b2,b3,b4,b5,b6,b7}, [%5], %7;\n"
#endif
                        "wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16.satfinite "
                        "    {%0,%1,%2,%3},\n"
                        "    {a0,a1,a2,a3,a4,a5,a6,a7},\n"
                        "    {b0,b1,b2,b3,b4,b5,b6,b7},\n"
                        "    {%8,%9,%10,%11};\n"
                        "}": "=r"(d0_), "=r"(d1_), "=r"(d2_), "=r"(d3_) : "l"(bb_agm_), "l"(bb_bgm_), "r"(agm_stride), "r"(bgm_stride), "r"(c0_), "r"(c1_), "r"(c2_), "r"(c3_));
                    c0[mb][nb] = d0_;
                    c1[mb][nb] = d1_;
                    c2[mb][nb] = d2_;
                    c3[mb][nb] = d3_;
#else
                    for(m = offset_m; m < 16; m += 8) {
                        for(n = offset_n; n < 16; n += 4) {
                            float a = 0.0f;
                            for(k = 0; k < 16; k++) {
                                a += vload_half(agm_stride * k + m, bb_agm_) * vload_half(bgm_stride * k + n, bb_bgm_);
                            }
                            acc[mb][nb][m/8][n/4] += a;
                        }
                    }
#endif
                }
            }
        }
    }

#ifdef USE_TC
#pragma unroll
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
#pragma unroll
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            int c0_ = c0[mb][nb];
            int c1_ = c1[mb][nb];
            int c2_ = c2[mb][nb];
            int c3_ = c3[mb][nb];
            __global half * b_cgm_ = cgm_ + kSizeM * nb * 16 + mb * 16;
            asm("{\n"
                "wmma.store.d.sync.aligned.global.col.m16n16k16.f16 [%4], {%0,%1,%2,%3}, %5;"
                "}" : : "r"(c0_), "r"(c1_), "r"(c2_), "r"(c3_), "l"(b_cgm_), "r"(kSizeM));
        }
    }
#else
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            for(m = offset_m; m < 16; m += 8) {
                for(n = offset_n; n < 16; n += 4) {
                    vstore_half(acc[mb][nb][m/8][n/4], kSizeM * (nb * 16 + n) + mb * 16 + m, cgm_);
                }
            }
        }
    }
#endif
}

struct alm_t {short alm[KWG * MWG];} __attribute__((aligned(32)));
struct blm_t {short blm[KWG * NWG];} __attribute__((aligned(32)));

__kernel __attribute__((reqd_work_group_size(32*MDIMC/16, NDIMC/16, 1)))
void XgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                  const __global half* restrict agm,
                  const __global half* restrict bgm,
                  __global half* restrict cgm)
{
    // Sets the offsets
    const int batch = get_group_id(2);
    const int a_offset = kSizeM*kSizeK*batch;
    const int b_offset = kSizeK*kSizeN*batch;
    const int c_offset = kSizeM*kSizeN*batch;

    const __global half* restrict agm_ = &agm[a_offset];
    const __global half* restrict bgm_ = &bgm[b_offset];
    __global half* restrict cgm_ = &cgm[c_offset];

    // Allocates workgroup-private memory (local memory)
    #if SA == 1
      __local struct alm_t alm;
    #endif
    #if SB == 1
      __local struct blm_t blm;
    #endif

    #if SA == 1 && SB == 1
        HgemmBody(kSizeM, kSizeN, kSizeK, alm.alm, blm.blm, agm_, bgm_, cgm_);
    #elif SA == 1
        HgemmBody(kSizeM, kSizeN, kSizeK, alm.alm, agm_, bgm_, cgm_);
    #elif SB == 1
        HgemmBody(kSizeM, kSizeN, kSizeK, blm.blm, agm_, bgm_, cgm_);
    #else
        HgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_);
    #endif
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
