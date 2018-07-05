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
    class ContextPoolEntry {
    public:
        size_t net_index;
        OpenCLContext context;
        ContextPoolEntry(size_t index) : net_index(index) {}
    };
public:
    void initialize(const int channels);
    std::vector<std::unique_ptr<OpenCL_Network>> & get_networks() {
        return m_networks;
    }
    void forward(const std::vector<net_t>& input,
                 std::vector<net_t>& output_pol,
                 std::vector<net_t>& output_val);
private:
    std::vector<std::unique_ptr<OpenCL_Network>> m_networks;
    std::vector<std::unique_ptr<OpenCL>> m_opencl;

    using ContextPoolQueue = std::list<std::shared_ptr<ContextPoolEntry>>;
    std::vector<ContextPoolQueue> m_context_pool;

    std::mutex m_context_pool_lock;
    std::condition_variable m_context_pool_condvar;
};

#endif
