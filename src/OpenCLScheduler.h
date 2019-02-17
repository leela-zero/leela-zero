/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors

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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef OPENCLSCHEDULER_H_INCLUDED
#define OPENCLSCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <vector>
#include <thread>

#include "SMP.h"
#include "ForwardPipe.h"
#include "OpenCL.h"
#include "ThreadPool.h"

#ifndef NDEBUG
struct batch_stats_t {
    std::atomic<size_t> single_evals{0};
    std::atomic<size_t> batch_evals{0};
};
extern batch_stats_t batch_stats;
#endif

template <typename net_t>
class OpenCLScheduler : public ForwardPipe {
    class ForwardQueueEntry {
    public:
        std::mutex mutex;
        std::condition_variable cv;
        const std::vector<float>& in;
        std::vector<float>& out_p;
        std::vector<float>& out_v;
        ForwardQueueEntry(const std::vector<float>& input,
                          std::vector<float>& output_pol,
                          std::vector<float>& output_val)
        : in(input), out_p(output_pol), out_v(output_val)
          {}
    };
public:
    virtual ~OpenCLScheduler();
    OpenCLScheduler();

    virtual void initialize(const int channels);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);
    virtual bool needs_autodetect();
    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights);
private:
    bool m_running = true;
    std::vector<std::unique_ptr<OpenCL_Network<net_t>>> m_networks;
    std::vector<std::unique_ptr<OpenCL<net_t>>> m_opencl;

    std::mutex m_mutex;
    std::condition_variable m_cv;

    // start with 10 milliseconds : lock protected
    int m_waittime{10};

    // set to true when single (non-batch) eval is in progress
    std::atomic<bool> m_single_eval_in_progress{false};

    std::list<std::shared_ptr<ForwardQueueEntry>> m_forward_queue;
    std::list<std::thread> m_worker_threads;

    void batch_worker(const size_t gnum);
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

    void push_convolve(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights);
};

#endif
