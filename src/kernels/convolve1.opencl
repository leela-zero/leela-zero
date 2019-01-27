/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(
    __kernel
    __attribute__((work_group_size_hint(8, 16, 1)))
    void convolve1(
                   __global const net_t * restrict in,
                   __global net_t * restrict merge,
                   __global const net_t * restrict weights,
                   __local real * channel_buff,
                   __local real * row_buff) {

        // cl::NDRange global(channels, outputs, row);
        const int c   = get_global_id(0);  // channel
        const int o   = get_global_id(1);  // output
        const int row_batch = get_global_id(2);  // row * batch_size

        const int row = row_batch % BOARD_SIZE;
        const int batch = row_batch / BOARD_SIZE;

        const int channels = get_global_size(0);
        const int outputs  = get_global_size(1);

        const int input_offset = batch * NUM_INTERSECTIONS * channels;
        const int merge_offset = batch * NUM_INTERSECTIONS * (channels >> 3) * outputs;

        // cl::NDRange local(2, (1->32), 1);
        const int lx = get_local_id(0);
        const int ly = get_local_id(1);
        const int chan_buff_size = 8;
        const int out_buff_size  = get_local_size(1);
        const int row_buff_size  = 7;
        const int chan_shift     = 3;
        // input = channels * height * width
        // output = outputs * height * width
        // weights = output * channels * filter
        // merge = channels * outputs * height * width
        const int width = BOARD_SIZE;
        const int height = BOARD_SIZE;
        const int strip_size = width;
        // Copy the input channels (strips) locally
        if (out_buff_size < BOARD_SIZE && ly == 0) {
            // strip-row
            for (int w = 0; w < width; w++) {
                channel_buff[lx * width + w] =
                    vload_net_t((c * height + row) * width + w + input_offset, in);
            }
        } else if (out_buff_size >= BOARD_SIZE && ly < BOARD_SIZE) {
            // Every thread copies a column
            channel_buff[lx * width + ly] = vload_net_t((c * height + row) * width +
                ly + input_offset, in);
        }
        // Copy the filter we are applying locally
        __private real filter_buff = vload_net_t((o * channels + c), weights);
        barrier(CLK_LOCAL_MEM_FENCE);
        int out_lane = 0;
        int out_cw   = 0;
        #pragma unroll
        for (int cw = 0; cw < width; cw++) {
            int fid = lx * strip_size;
            real out  = channel_buff[fid + cw] * filter_buff;
            row_buff[(ly * chan_buff_size + lx) * row_buff_size + out_lane] = out;
            out_lane++;
            // Row buffer full or last lane?
            if (out_lane == row_buff_size || (cw == width - 1)) {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (lx < out_lane) {
                    real val;
                    val  = row_buff[(ly * chan_buff_size + 0) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 1) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 2) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 3) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 4) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 5) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 6) * row_buff_size + lx];
                    val += row_buff[(ly * chan_buff_size + 7) * row_buff_size + lx];
                    vstore_net_t(val, (((c >> chan_shift) * height + row) * width +
                        out_cw + lx) * outputs + o + merge_offset, merge);
                }
                out_cw  += row_buff_size;
                out_lane = 0;
           }
       }
    }

__kernel void merge(
                        __global const net_t * restrict in,
                        __global net_t * restrict out,
                        __private const int channels) {
        // cl::NDRange global(outputs, NUM_INTERSECTIONS);
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        const int batch = get_global_id(2);
        const int output = gx;
        const int b = gy;
        const int outputs = get_global_size(0);
        const int width = BOARD_SIZE;
        const int height = BOARD_SIZE;
        const int o = output;
        real sum = 0;
        for (int c = 0; c < channels; c++) {
            sum += vload_net_t(batch * channels * NUM_INTERSECTIONS * outputs +
                (c * NUM_INTERSECTIONS + b) * outputs + o, in);
        }
        vstore_net_t(sum, batch * outputs * NUM_INTERSECTIONS + o * NUM_INTERSECTIONS + b, out);
    }

// End of the C++11 raw string literal
)"
