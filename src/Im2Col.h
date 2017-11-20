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

#include <vector>

template <unsigned long filter_size>
void im2col(const int channels,
            const std::vector<float>& input,
            std::vector<float>& output) {
    // Not implemented
    // See github.com/gcp/leela-zero/pull/104 for first pass implementation.
    exit(EXIT_FAILURE);
}

template <>
void im2col<1>(const int channels,
            const std::vector<float>& input,
            std::vector<float>& output) {
    constexpr unsigned int boardsize = 19;
    unsigned int outSize = channels * boardsize * boardsize;
    assert(output.size() == outSize);

    std::copy(input.begin(), input.begin() + outSize, output.data());
}

#endif
