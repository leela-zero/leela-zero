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

//TODO: __constant define

__device__ real Bt[WINOGRAD_ALPHA * WINOGRAD_ALPHA] = \
                   {1.0f,  0.0f,     -5.0f/2.0f,  0.0f,      1.0f, 0.0f,
                    0.0f, -SQ2,      -2.0f,       SQ2/2.0f,  1.0f, 0.0f,
                    0.0f,  SQ2,      -2.0f,      -SQ2/2.0f,  1.0f, 0.0f,
                    0.0f, -SQ2/2.0f, -1.0f/2.0f,  SQ2,       1.0f, 0.0f,
                    0.0f,  SQ2/2.0f, -1.0f/2.0f, -SQ2,       1.0f, 0.0f,
                    0.0f,  1.0f,      0.0f,      -5.0f/2.0f, 0.0f, 1.0f};

__device__ real At[WINOGRAD_M * WINOGRAD_ALPHA] = \
                   {1.0f, 1.0f,      1.0f,       1.0f,      1.0f,     0.0f,
                    0.0f, SQ2/2.0f, -SQ2/2.0f,   SQ2,      -SQ2,      0.0f,
                    0.0f, 1.0f/2.0f, 1.0f/2.0f,  2.0f,      2.0f,     0.0f,
                    0.0f, SQ2/4.0f, -SQ2/4.0f,   2.0f*SQ2, -2.0f*SQ2, 1.0f};

__device__ void __in_transform_eq(real x[WINOGRAD_ALPHA][WINOGRAD_ALPHA], __global net_t * restrict V, int offset, int CPpad) {

    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int P = WTILES * WTILES;

    real T1[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
    real T2[WINOGRAD_ALPHA][WINOGRAD_ALPHA];

    // Calculates transpose(B).x.B
    for (int i = 0; i < WINOGRAD_ALPHA; i++){
        for (int j = 0; j < WINOGRAD_ALPHA; j++) {
#ifdef WINOGRAD_SIMD
            real2 acc = {ZERO, ZERO};
            real2 *x2 = (real2 *)&x[j][0];
            for (int k = 0; k < WINOGRAD_ALPHA/2; k++) {
                real2 x1;
                x1.x = Bt[i * WINOGRAD_ALPHA + 2*k];
                x1.y = Bt[i * WINOGRAD_ALPHA + 2*k + 1];
                acc += x1 * x2[k];
            }
            T1[i][j] = acc.x + acc.y;
#else
            real acc = ZERO;
            for (int k = 0; k < WINOGRAD_ALPHA; k++) {
                acc += Bt[i * WINOGRAD_ALPHA + k] * x[j][k];
            }
            T1[i][j] = acc;
#endif
        }
    }

    for (int i = 0; i < WINOGRAD_ALPHA; i++){
        for (int j = 0; j < WINOGRAD_ALPHA; j++) {
#ifdef WINOGRAD_SIMD
            real2 acc = {ZERO, ZERO};
            real2 *x1 = (real2 *)&T1[i][0];
            for (int k = 0; k < WINOGRAD_ALPHA/2; k++) {
                real2 x2;
                x2.x = Bt[j * WINOGRAD_ALPHA + 2*k];
                x2.y = Bt[j * WINOGRAD_ALPHA + 2*k + 1];
                acc += x1[k] * x2;
            }
            T2[i][j] = acc.x + acc.y;
#else
            real acc = ZERO;
            for (int k = 0; k < WINOGRAD_ALPHA; k++) {
                acc += T1[i][k] * Bt[j * WINOGRAD_ALPHA + k];
            }
            T2[i][j] = acc;
#endif
        }
    }

    // Scatter each sub element in tile to separate matrices
    for (int i = 0; i < WINOGRAD_ALPHA; i++) {
        for (int j = 0; j < WINOGRAD_ALPHA; j++) {
            vstore_net_t(T2[i][j], (i*WINOGRAD_ALPHA + j)*CPpad + offset, V);
        }
    }
}

__kernel void in_transform(__global net_t * restrict in, __global net_t * restrict V,
                           const int C, const int Cpad,
                           const int Ppad, const int batch_size) {
    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int P = WTILES * WTILES;
    const int CPpad = Ppad * Cpad;

    const int block = get_global_id(0);
    const int ch = get_global_id(1);

    const int batch = block / P;
    const int block_x = (block - P * batch) % WTILES;
    const int block_y = (block - P * batch) / WTILES;

    // 6x6 tiles overlap by 2
    const int yin = WINOGRAD_M * block_y - 1;
    const int xin = WINOGRAD_M * block_x - 1;

    if (block < batch_size * P && ch < C) {
        // Cache input tile and handle zero padding
        real x[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
        for (int i = 0; i < WINOGRAD_ALPHA; i++) {
            for (int j = 0; j < WINOGRAD_ALPHA; j++) {
                int a = xin + j;
                int b = yin + i;
                // x is transposed here for better layout later
                if (b >= 0 && a >= 0 && b < H && a < W) {
                    x[j][i] = vload_net_t(batch * C * NUM_INTERSECTIONS +
                        ch * NUM_INTERSECTIONS + b * W + a, in);
                } else {
                    x[j][i] = ZERO;
                }
            }
        }

        // V dimensions are [36, input_channels, batch_size * tiles].
        // Padded with zeros as necessary for SGEMM
        // = [36, Cpad, Ppad]

        const int offset = ch * Ppad + block;
        __in_transform_eq(x, V, offset, CPpad);
    }
}

__device__ void __out_transform_eq(__global const net_t * restrict M, real o[WINOGRAD_M * WINOGRAD_M],
                        int Kpad, int Ppad, int block)
{

    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int P = WTILES * WTILES;

    const int k = get_global_id(0);
    real temp_m[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
    real temp[WINOGRAD_M][WINOGRAD_ALPHA];


    // M dimensions are [36, outputs, batch_size * tiles].
    // Plus zero padding from SGEMM.

    const int offset = block * Kpad + k;

    for (int yn = 0; yn < WINOGRAD_ALPHA; yn++) {
        for (int xn = 0; xn < WINOGRAD_ALPHA; xn++) {
            temp_m[xn][yn] = vload_net_t((yn * WINOGRAD_ALPHA + xn) * Kpad * Ppad + offset, M);
        }
    }

    // Calculates transpose(A).temp_m.A
    for (int i = 0; i < WINOGRAD_M; i++){
        for (int j = 0; j < WINOGRAD_ALPHA; j++) {
#ifdef WINOGRAD_SIMD
            real2 acc = {ZERO, ZERO};
            real2 *x2 = (real2 *)&temp_m[j][0];
            for (int q = 0; q < WINOGRAD_ALPHA/2; q++) {
                real2 x1;
                x1.x = At[i * WINOGRAD_ALPHA + 2*q];
                x1.y = At[i * WINOGRAD_ALPHA + 2*q + 1];
                acc += x1 * x2[q];
            }
            temp[i][j] = acc.x + acc.y;
#else
            real acc = ZERO;
            for (int q = 0; q < WINOGRAD_ALPHA; q++) {
                acc += At[i * WINOGRAD_ALPHA + q] * temp_m[j][q];
            }
            temp[i][j] = acc;
#endif
        }
    }

    for (int i = 0; i < WINOGRAD_M; i++){
        for (int j = 0; j < WINOGRAD_M; j++) {
#ifdef WINOGRAD_SIMD
            real2 acc = {ZERO, ZERO};
            real2 *x1 = (real2 *)&temp[i][0];
            for (int q = 0; q < WINOGRAD_ALPHA/2; q++) {
                real2 x2;
                x2.x = At[j * WINOGRAD_ALPHA + 2*q];
                x2.y = At[j * WINOGRAD_ALPHA + 2*q + 1];
                acc += x1[q] * x2;
            }
            o[i * WINOGRAD_M + j] = acc.x + acc.y;
#else
            real acc = ZERO;
            for (int q = 0; q < WINOGRAD_ALPHA; q++) {
                acc += temp[i][q] * At[j * WINOGRAD_ALPHA + q];
            }
            o[i * WINOGRAD_M + j] = acc;
#endif
        }
    }
}

__kernel void out_transform_fused_bn(__global const net_t * restrict M,
                                     __global net_t * restrict Y,
                                     const int K,
                                     const int Kpad, const int Ppad,
                                     const int batch_size,
                                     __global const net_t * restrict residual,
                                     __global __constant net_t * restrict means,
                                     __global __constant net_t * restrict stddivs) {

    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int P = WTILES * WTILES;

    const int k = get_global_id(0);
    const int block = get_global_id(1);

    const int batch = block / P;
    const int block_x = (block - P * batch) % WTILES;
    const int block_y = (block - P * batch) / WTILES;

    int x = WINOGRAD_M * block_x;
    int y = WINOGRAD_M * block_y;

    if (k < K && block < batch_size * P) {
        const int kHW = batch * K * NUM_INTERSECTIONS + k * NUM_INTERSECTIONS;

        real o[WINOGRAD_M * WINOGRAD_M];
        __out_transform_eq(M, o, Kpad, Ppad, block);

        const real mean = vload_net_t(k, means);
        const real scale_stddiv = vload_net_t(k, stddivs);

        for (int i = 0; i < WINOGRAD_M; i++) {
            for (int j = 0; j < WINOGRAD_M; j++) {
                const int in_idx = i * WINOGRAD_M + j;
                const int out_idx = (y + i) * W + (x + j);
                if (y + i < H && x + j < W) {
                    o[in_idx] = scale_stddiv * (o[in_idx] - mean);
                    if (residual) {
                        o[in_idx] += vload_net_t(kHW + out_idx, residual);
                    }
                    o[in_idx] = o[in_idx] > 0 ? o[in_idx] : ZERO;
                    vstore_net_t(o[in_idx], kHW + out_idx, Y);
                }
            }
        }
    }
}

__kernel void out_transform_fused_bn_in(
                                     __global const net_t * restrict M,
                                     __global net_t * restrict Y,
                                     __global net_t * restrict V,
                                     const int K,
                                     const int Kpad, const int Ppad, const int Cpad,
                                     __global const net_t * restrict residual,
                                     __constant const net_t * restrict means,
                                     __constant const net_t * restrict stddivs) {

    const int W = BOARD_SIZE;
    const int H = BOARD_SIZE;
    const int P = WTILES * WTILES;

    const int k = get_global_id(0);
    const int kg = get_local_id(0);
    const int block = get_global_id(1);
    const int batch = get_global_id(2);

    const int block_x = block % WTILES;
    const int block_y = block / WTILES;

    const int x = WINOGRAD_M * block_x;
    const int y = WINOGRAD_M * block_y;

    const int kHW = batch * K * NUM_INTERSECTIONS + k * NUM_INTERSECTIONS;

    __local real ybuf[OUTIN_KWG * NUM_INTERSECTIONS];

    if (k < K && block < P) {

        const real mean = vload_net_t(k, means);
        const real scale_stddiv = vload_net_t(k, stddivs);

        real temp_m[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
        real Atp[WINOGRAD_M * WINOGRAD_ALPHA];
        real temp[WINOGRAD_ALPHA];

        // M dimensions are [36, outputs, batch_size * tiles].
        // Plus zero padding from SGEMM.

        const int offset = block * Kpad + k;

        for (int i = 0; i < WINOGRAD_M * WINOGRAD_ALPHA; i++) {
            Atp[i] = At[i];
        }

        for (int yn = 0; yn < WINOGRAD_ALPHA; yn++) {
            for (int xn = 0; xn < WINOGRAD_ALPHA; xn++) {
                temp_m[xn][yn] = vload_net_t((yn * WINOGRAD_ALPHA + xn) * Kpad * Ppad + offset, M);
            }
        }

        // Calculates transpose(A).temp_m.A
        for (int i = 0; i < WINOGRAD_M; i++) {
            for (int j = 0; j < WINOGRAD_ALPHA; j++) {
                real acc = ZERO;
                for (int q = 0; q < WINOGRAD_ALPHA; q++) {
                    acc += Atp[i * WINOGRAD_ALPHA + q] * temp_m[j][q];
                }
                temp[j] = acc;
            }

            for (int j = 0; j < WINOGRAD_M; j++) {
                real acc = ZERO;
                for (int q = 0; q < WINOGRAD_ALPHA; q++) {
                    acc += temp[q] * Atp[j * WINOGRAD_ALPHA + q];
                }

                const int out_idx = (y + i) * W + (x + j);
                acc = scale_stddiv * (acc - mean);
                if (y + i < H && x + j < W) {
                    ybuf[kg * NUM_INTERSECTIONS + out_idx] = acc;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int ks = get_local_size(0);
    const int k0 = get_group_id(0) * get_local_size(0);

    for (int x = get_local_id(0) + ks * get_local_id(1); x < ks * NUM_INTERSECTIONS; x += get_local_size(1) * get_local_size(0)) {
        const int kx = x / NUM_INTERSECTIONS;
        const int idx = x - kx * NUM_INTERSECTIONS;

        const int kHWx = batch * K * NUM_INTERSECTIONS + (k0 + kx) * NUM_INTERSECTIONS;

        real acc = ybuf[kx * NUM_INTERSECTIONS + idx];
        if (residual) {
            acc += vload_net_t(kHWx + idx, residual);
        }
        acc = max(acc, ZERO);

        if (Y) {
            vstore_net_t(acc, kHWx + idx, Y);
        }
        ybuf[kx * NUM_INTERSECTIONS + idx] = acc;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int yin = WINOGRAD_M * block_y - 1;
    const int xin = WINOGRAD_M * block_x - 1;

    if (block < P && k < K) {
        const int CPpad = Ppad * Cpad;
        // Cache input tile and handle zero padding
        real xx[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
        for (int i = 0; i < WINOGRAD_ALPHA; i++) {
            int b = yin + i;
            for (int j = 0; j < WINOGRAD_ALPHA; j++) {
                int a = xin + j;
                // x is transposed here for better layout later
                if (b >= 0 && a >= 0 && b < H && a < W) {
                    xx[j][i] = ybuf[kg * NUM_INTERSECTIONS + b * W + a];
                } else {
                    xx[j][i] = ZERO;
                }
            }
        }

        const int offset = k * Ppad + P * batch + block;
        __in_transform_eq(xx, V, offset, CPpad);
    }
}

