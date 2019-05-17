/*
    This file is part of Leela Zero.
    Copyright (C) 2019 Henrik Forsten and contributors

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

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
    __kernel void global_avg_pooling(
                   const int channels,
                   __global const net_t * restrict in,
                   __global net_t * restrict out) {

        const int col = get_global_id(0);  // column
        const int c = get_global_id(1);  // channel

        const int lid = get_local_id(0);

        __local real row_acc[BOARD_SIZE];

        if (c < channels && col < BOARD_SIZE) {

            real acc = ZERO;

            for ( int i = 0; i < BOARD_SIZE; i++) {
                acc += vload_net_t(c * NUM_INTERSECTIONS + i * BOARD_SIZE + col, in);
            }
            row_acc[lid] = acc;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            real acc = ZERO;
            for ( int i = 0; i < BOARD_SIZE; i++) {
                acc += row_acc[i];
            }
            acc = acc/NUM_INTERSECTIONS;
            vstore_net_t(acc, c, out);
        }
    }
// End of the C++11 raw string literal
)"
