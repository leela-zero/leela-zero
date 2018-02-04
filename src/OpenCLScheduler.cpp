/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Junhee Yoo

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
#include "config.h"

#ifdef USE_OPENCL
#include "GTP.h"
#include "Random.h"
#include "OpenCLScheduler.h"

thread_local auto current_thread_gpu_num = size_t{0};
OpenCLScheduler opencl;

void OpenCLScheduler::initialize(const int channels) {
    // multi-gpu?
    if (!cfg_gpus.empty()) {
        auto silent{false};
        for(auto gpu : cfg_gpus) {
            auto opencl = std::make_unique<OpenCL>();
            auto net = std::make_unique<OpenCL_Network>(*opencl);
            opencl->initialize(channels, {gpu}, silent);
            m_opencl.push_back(std::move(opencl));
            m_networks.push_back(std::move(net));

            // Clear thread data on every init call.  We don't know which GPU
            // this thread will be eventually be assigned to
            opencl_thread_data = ThreadData();

            // starting next GPU, let's not dump full list of GPUs
            silent = true;
        }

        for(size_t gnum = 0; gnum < m_networks.size(); gnum++) {
            // launch the worker thread.  2 threads so that we can fully
            // utilize GPU, since the worker thread consists of some CPU
            // work for task preparation.
            constexpr auto num_threads = 2;
            for(auto i = 0; i < num_threads; i++) {
                m_threadpool.add_thread([gnum] {
                    current_thread_gpu_num = gnum;
                });
            }
        }
    } else {
        auto opencl = std::make_unique<OpenCL>();
        auto net = std::make_unique<OpenCL_Network>(*opencl);
        opencl->initialize(channels, {});

        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));
    }
}

void OpenCLScheduler::forward(const std::vector<net_t>& input,
                              std::vector<net_t>& output_pol,
                              std::vector<net_t>& output_val) {
    if (m_networks.size() == 1) {
        m_networks[0]->forward(input, output_pol, output_val);
        return;
    }

    auto f = m_threadpool.add_task([this, &input, &output_pol, &output_val]{
        m_networks[current_thread_gpu_num]->forward(input, output_pol, output_val);
    });

    f.get();
}
#endif
