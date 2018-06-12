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
        auto gnum = size_t{0};
        for (auto gpu : cfg_gpus) {
            auto opencl = std::make_unique<OpenCL>();
            auto net = std::make_unique<OpenCL_Network>(*opencl);
            opencl->initialize(channels, {gpu}, silent);
            m_opencl.push_back(std::move(opencl));
            m_networks.push_back(std::move(net));

            // starting next GPU, let's not dump full list of GPUs
            silent = true;

            // launch the worker thread.  2 threads so that we can fully
            // utilize GPU, since the worker thread consists of some CPU
            // work for task preparation.
            constexpr auto num_threads = 2;
            for (auto i = 0; i < num_threads; i++) {
                OpenCLContext ctx;
                ctx.gpu_num = gnum;
                ctx.opencl_thread_data = std::make_unique<ThreadData>();
                m_context.push_back(std::move(ctx));
            }
            gnum++;
        }
    } else {
        auto opencl = std::make_unique<OpenCL>();
        auto net = std::make_unique<OpenCL_Network>(*opencl);
        opencl->initialize(channels, {});

        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));
    
        // launch the worker thread.  2 threads so that we can fully
        // utilize GPU, since the worker thread consists of some CPU
        // work for task preparation.
        constexpr auto num_threads = 2;
        for (auto i = 0; i < num_threads; i++) {
            OpenCLContext ctx;
            ctx.gpu_num = 0;
            ctx.opencl_thread_data = std::make_unique<ThreadData>();
            m_context.push_back(std::move(ctx));
        }
    }
}

void OpenCLScheduler::forward(const std::vector<net_t>& input,
                              std::vector<net_t>& output_pol,
                              std::vector<net_t>& output_val) {
    OpenCLContext ctx;

    {
        std::unique_lock<std::mutex> lk(m_context_lock);
        m_context_condvar.wait(lk, [this]{return !m_context.empty();});
        ctx = std::move(m_context.front());
        m_context.pop_front();
    }
  
    m_networks[ctx.gpu_num]->forward(input, output_pol, output_val, *ctx.opencl_thread_data);
 
    {
        std::unique_lock<std::mutex> lk(m_context_lock);
        m_context.push_back(std::move(ctx));
        lk.unlock();
        m_context_condvar.notify_one();
    }
}
#endif
