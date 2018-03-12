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
#include <cassert>

#include "config.h"

#ifdef USE_OPENCL
#include "GTP.h"
#include "Random.h"
#include "OpenCLScheduler.h"

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
            if (opencl_thread_data != nullptr) {
                opencl_thread_data = nullptr;
            }

            // starting next GPU, let's not dump full list of GPUs
            silent = true;
        }

        for(size_t gnum = 0; gnum < m_networks.size(); gnum++) {
            // two instances for optimal scalability
            m_thread_data.emplace_back( new ThreadData(gnum) );
            m_thread_data.emplace_back( new ThreadData(gnum) );
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

    // On multi-gpu situations you shouldn't have a thread context to start with.
    assert(opencl_thread_data == nullptr);

    // acquire a thread context.
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        if (m_thread_data.empty()) {
            m_cv.wait(lk, [this]() { return !m_thread_data.empty(); });
        }
        
        assert(!m_thread_data.empty());
        opencl_thread_data = std::move(m_thread_data.front());
        m_thread_data.pop_front();
    }

    // run it.
    m_networks[opencl_thread_data->m_gpu_num]->forward(input, output_pol, output_val);

    // ...and release the thread context
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_thread_data.push_back(std::move(opencl_thread_data));
        m_cv.notify_one();
    }
}
#endif
