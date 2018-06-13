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
    auto gpus = cfg_gpus;
    if (gpus.empty()) {
        gpus = {0};
    }

    auto silent{false};
    auto gnum = size_t{0};

    // launch the worker thread.  2 threads so that we can fully
    // utilize GPU, since the worker thread consists of some CPU
    // work for task preparation.
    constexpr auto num_threads = 2;

    for (auto i = 0; i < num_threads; i++) {
        m_context.emplace_back();
    }

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL>();
        auto net = std::make_unique<OpenCL_Network>(*opencl);
        opencl->initialize(channels, {gpu}, silent);
        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));

        // starting next GPU, let's not dump full list of GPUs
        silent = true;

        for (auto i = 0; i < num_threads; i++) {
            OpenCLContext ctx;
            ctx.gpu_num = gnum;
            ctx.opencl_thread_data = std::make_unique<ThreadData>();
            m_context[i].push_back(std::move(ctx));
        }
        gnum++;
    }
}

void OpenCLScheduler::forward(const std::vector<net_t>& input,
                              std::vector<net_t>& output_pol,
                              std::vector<net_t>& output_val) {
    OpenCLContext ctx;
    auto queue_num = size_t{0};
    {
        std::unique_lock<std::mutex> lk(m_context_lock);
        m_context_condvar.wait(lk, [this]{
            for(auto & ctx : m_context) {
                if (!ctx.empty()) {
                    return true;
                }
            }
            return false;
        });
        while (queue_num < m_context.size()) {
            if(!m_context[queue_num].empty()) {
                ctx = std::move(m_context[queue_num].front());
                m_context[queue_num].pop_front();
                break;
            }
            queue_num++;
        }
    }
  
    m_networks[ctx.gpu_num]->forward(input, output_pol, output_val, *ctx.opencl_thread_data);
 
    {
        std::unique_lock<std::mutex> lk(m_context_lock);
        m_context[queue_num].push_back(std::move(ctx));
        lk.unlock();
        m_context_condvar.notify_one();
    }
}
#endif
