/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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

#ifndef IM2COL_H_INCLUDED
#define IM2COL_H_INCLUDED

#include "config.h"
#include <vector>
#include <algorithm>
#include "Utils.h"

template <unsigned long filter_size>
void im2col(const int channels,
            const std::vector<float>& input,
            std::vector<float>& output) {
    constexpr unsigned int height = 19;
    constexpr unsigned int width = 19;
    constexpr unsigned int channel_size = height * width;

    constexpr int pad = (filter_size / 2);
    constexpr unsigned int output_h = height + 2 * pad - filter_size  + 1;
    constexpr unsigned int output_w = width + 2 * pad - filter_size + 1;

    const float* data_im = input.data();
    float* data_col = output.data();

    for (int channel = 0; channel < channels; channel++, data_im += channel_size) {
        for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (unsigned int kernel_col = 0; kernel_col < filter_size; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    int input_col = -pad + kernel_col;
                    for (int output_col = output_w; output_col; output_col--) {
                        if (input_row > 0 && input_row < height &&
                            input_col > 0 && input_col < width) {
                            *(data_col++) = data_im[input_row * width + input_col];
                        } else {
                            *(data_col++) = 0;
                        }
                        input_col++;
                    }
                    input_row++;
                }
            }
        }
    }
}


#endif
