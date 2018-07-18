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


void OpenCLScheduler::initialize(const int channels) {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // an empty GPU list from the command line represents autodetect.
    // put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};
    auto gnum = size_t{0};

    // launch the worker thread.  2 threads so that we can fully
    // utilize GPU, since the worker thread consists of some CPU
    // work for task preparation.
    constexpr auto num_threads = 2;
    m_context_pool.resize(num_threads);

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL>();
        auto net = std::make_unique<OpenCL_Network>(*opencl);
        opencl->initialize(channels, gpu, silent);
        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));

        // starting next GPU, let's not dump full list of GPUs
        silent = true;

        for (auto i = 0; i < num_threads; i++) {
            m_context_pool[i].emplace_back(std::make_shared<ContextPoolEntry>(gnum));
        }
        gnum++;
    }
}

void OpenCLScheduler::forward(const std::vector<net_t>& input,
                              std::vector<net_t>& output_pol,
                              std::vector<net_t>& output_val) {
    std::shared_ptr<ContextPoolEntry> ctx;
    auto queue_num = size_t{0};
    {
        std::unique_lock<std::mutex> lk(m_context_pool_lock);
        m_context_pool_condvar.wait(lk, [this]{
            for (auto & ctxlist : m_context_pool) {
                if (!ctxlist.empty()) {
                    return true;
                }
            }
            return false;
        });
        while (queue_num < m_context_pool.size()) {
            if (!m_context_pool[queue_num].empty()) {
                ctx = std::move(m_context_pool[queue_num].front());
                m_context_pool[queue_num].pop_front();
                break;
            }
            queue_num++;
        }
        // if this failed, it means the condition variable exited itself
        // when the predicate condition return false
        assert(ctx != nullptr);
    }

    m_networks[ctx->net_index]->forward(input, output_pol, output_val, ctx->context);

    {
        std::unique_lock<std::mutex> lk(m_context_pool_lock);
        m_context_pool[queue_num].push_back(std::move(ctx));
        lk.unlock();
        m_context_pool_condvar.notify_one();
    }
}
#endif
