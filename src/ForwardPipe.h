/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Junhee Yoo and contributors

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

#ifndef FORWARDPIPE_H_INCLUDED
#define FORWARDPIPE_H_INCLUDED

#include <vector>

#include "config.h"

class ForwardPipe {
public:
    virtual ~ForwardPipe() = default;

    virtual void initialize(const int channels) = 0;
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) = 0;

    virtual void push_input_convolution(unsigned int filter_size,
                                        unsigned int channels,
                                        unsigned int outputs,
                                        const std::vector<float>& weights,
                                        const std::vector<float>& means,
                                        const std::vector<float>& variances) = 0;

    virtual void push_residual(unsigned int filter_size,
                               unsigned int channels,
                               unsigned int outputs,
                               const std::vector<float>& weights_1,
                               const std::vector<float>& means_1,
                               const std::vector<float>& variances_1,
                               const std::vector<float>& weights_2,
                               const std::vector<float>& means_2,
                               const std::vector<float>& variances_2) = 0;

    virtual void push_convolve(unsigned int filter_size,
                               unsigned int channels,
                               unsigned int outputs,
                               const std::vector<float>& weights) = 0;

    virtual void set_batching(bool is_batching) = 0;


};

#endif
