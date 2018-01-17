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
#include "GTP.h"
#include "Random.h"
#include "OpenCLScheduler.h"

void OpenCLScheduler::initialize(const int channels) {
    // multi-gpu?
    if(!cfg_gpus.empty()) {
        bool silent = false;
        for(auto gpu : cfg_gpus) {
            auto opencl = new OpenCL();
            auto net = new OpenCL_Network(opencl);
            m_opencl.emplace_back(opencl);
            m_networks.emplace_back(net);
            opencl->initialize(channels, {gpu}, silent);
            silent = true;
           
            // clear thread data on every init call.
            // We don't know which GPU this thread will be eventually be assigned to
            opencl_thread_data = ThreadData();
        }
    } else {
        auto opencl = new OpenCL();
        auto net = new OpenCL_Network(opencl);
        m_opencl.emplace_back(opencl);
        m_networks.emplace_back(net);
        opencl->initialize(channels, {});
    }
}

void OpenCLScheduler::forward(const std::vector<net_t>& input, std::vector<net_t>& output) {
    if(m_networks.size() == 1) {
        m_networks[0]->forward(input, output);
        return;
    }

    bool expval = false;
    if(m_workers_launched.compare_exchange_strong(expval, true)) {
        // test run each batch size.  pick ones that successfully passed sanity check
   
        for(auto & p : m_networks) {
            // launch the worker thread.  2 threads so that we can fully utilize GPU, since the 
            // worker thread consists of some CPU work for task preparation.
            constexpr int num_threads = 2;
            for(int i=0; i<num_threads; i++) {
                std::thread worker( [this, &p]{
                    std::unique_lock<std::mutex> lk(m_task_mutex);
                    while(true) {
                        m_task_cond.wait(lk, [this]{ return (m_task_queue.size() != 0); });

                        ForwardTask task = std::move(m_task_queue.front());
                        m_task_queue.pop_front();
                        lk.unlock();
        
                        p->forward(*task.input, *task.output);

                        task.prom.set_value();
                        lk.lock();
                    }
                });
        
                worker.detach();
            }
        }
    }

    // to let the worker thread do it, push the network evaluation into the task queue
    // and signal conditional variable that something is pushed.
    std::future<void> ret;
    {
        std::lock_guard<std::mutex> lock(m_task_mutex);
        ForwardTask tsk(&input, &output);
        ret = tsk.prom.get_future();
        m_task_queue.emplace_back(std::move(tsk));
    }
    m_task_cond.notify_one();

    // this method will return when worker thread finishes its job
    ret.get();

}

OpenCLScheduler opencl;

