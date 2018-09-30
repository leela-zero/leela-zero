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

#include <memory>
#include <vector>

#include "config.h"

class ForwardPipe {
public:
    class ForwardPipeWeights {
    public:
        // Input + residual block tower
        std::vector<std::vector<float>> m_conv_weights;
        std::vector<std::vector<float>> m_conv_biases;
        std::vector<std::vector<float>> m_batchnorm_means;
        std::vector<std::vector<float>> m_batchnorm_stddevs;

        // Policy head
        std::vector<float> m_conv_pol_w;
        std::vector<float> m_conv_pol_b;

        std::vector<float> m_conv_val_w;
        std::vector<float> m_conv_val_b;
    };

    virtual ~ForwardPipe() = default;

    virtual void initialize(const int channels) = 0;
    virtual bool needs_autodetect() { return false; };
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) = 0;
    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights) = 0;
};

#endif
