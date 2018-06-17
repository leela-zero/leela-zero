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

#ifndef OPENCL_SCHEDULER_H_INCLUDED
#define OPENCL_SCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <vector>
#include <future>

#include "OpenCL.h"
#include "ThreadPool.h"


class OpenCLScheduler {

private:
   

public:
    void initialize(const int channels);
    std::vector<std::unique_ptr<OpenCL_Network>> & get_networks() {
        return m_networks;
    }
    void forward(const std::vector<net_t>& input,
                 std::vector<net_t>& output_pol,
                 std::vector<net_t>& output_val);

    void push_input_convolution(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights,
                       const std::vector<float>& means,
                       const std::vector<float>& variances);

    void push_residual(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& means_1,
                       const std::vector<float>& variances_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& means_2,
                       const std::vector<float>& variances_2);

    void push_convolve1(unsigned int channels,
                        unsigned int outputs,
                        const std::vector<float>& weights);

private:
    std::vector<std::unique_ptr<OpenCL_Network>> m_networks;
    std::vector<std::unique_ptr<OpenCL>> m_opencl;
    // XXX : I failed to figure out how to make this compile with unique_ptr
    // especially on visual studio
    std::vector<std::list<std::shared_ptr<OpenCLContext>>> m_context;
    std::mutex m_context_lock;
    std::condition_variable m_context_condvar;
};

#endif
