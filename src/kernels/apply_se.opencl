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
    __kernel void apply_se(
                  const int channels,
                  const int batch_size,
                  __global const net_t * restrict input,
                  __global net_t * restrict residual,
                  __global const net_t * restrict fc_out) {

        const int col = get_global_id(0);  // column
        const int c = get_global_id(1);  // channel

        const int batch = c / channels;

        if (c < batch_size * channels && col < BOARD_SIZE) {
            real gamma = vload_net_t(c + batch * channels, fc_out);
            gamma = 1.0f/(1.0f + exp(-gamma)); // Sigmoid
            real beta = vload_net_t(c + batch * channels + channels, fc_out);

            for ( int i = 0; i < BOARD_SIZE; i++) {
                const int idx = c * NUM_INTERSECTIONS + i * BOARD_SIZE + col;
                const real in = vload_net_t(idx, input);
                const real res = vload_net_t(idx, residual);

                real val = gamma * in + res + beta;

                val = val > 0.0f ? val : 0.0f;

                vstore_net_t(val, idx, residual);
            }
        }
    }

// End of the C++11 raw string literal
)"
