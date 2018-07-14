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
#include "Network.h"
#include "Utils.h"
#include "OpenCLScheduler.h"

using Utils::ceilMultiple;

static std::vector<float> zeropad_U(const std::vector<float>& U,
                                      const int outputs, const int channels,
                                      const int outputs_pad,
                                      const int channels_pad) {
    // Fill with zeroes
    auto Upad = std::vector<float>(Network::WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto xi = 0; xi < Network::WINOGRAD_ALPHA; xi++){
                for (auto nu = 0; nu < Network::WINOGRAD_ALPHA; nu++) {
                    Upad[xi * (Network::WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (Network::WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

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
    m_context_pool.resize(num_threads);

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL>();
        auto net = std::make_unique<OpenCL_Network>(*opencl);
        opencl->initialize(channels, {gpu}, silent);
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

void OpenCLScheduler::push_input_convolution(unsigned int filter_size,
                   unsigned int channels,
                   unsigned int outputs,
                   const std::vector<float>& weights,
                   const std::vector<float>& means,
                   const std::vector<float>& variances) {

    for (const auto & opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(channels, kwg), vwm);

        const auto Upad = zeropad_U(weights,
                                    outputs, channels,
                                    m_ceil, k_ceil);
        opencl_net->push_input_convolution(
            filter_size, channels, outputs,
            Upad, means, variances
        );
    }
}

void OpenCLScheduler::push_residual(unsigned int filter_size,
                   unsigned int channels,
                   unsigned int outputs,
                   const std::vector<float>& weights_1,
                   const std::vector<float>& means_1,
                   const std::vector<float>& variances_1,
                   const std::vector<float>& weights_2,
                   const std::vector<float>& means_2,
                   const std::vector<float>& variances_2) {
    for (const auto & opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 = zeropad_U(weights_1,
                                     outputs, outputs,
                                     m_ceil, m_ceil);
        const auto Upad2 = zeropad_U(weights_2,
                                     outputs, outputs,
                                     m_ceil, m_ceil);
        opencl_net->push_residual(filter_size, channels, outputs,
                                  Upad1,
                                  means_1,
                                  variances_1,
                                  Upad2,
                                  means_2,
                                  variances_2);
    }
}

void OpenCLScheduler::push_convolve1(unsigned int channels,
                    unsigned int outputs,
                    const std::vector<float>& weights)
{
    for (const auto & opencl_net : m_networks) {
        opencl_net->push_convolve1(channels, outputs, weights);
    }
}

void OpenCLScheduler::forward(const std::vector<float>& input,
                              std::vector<float>& output_pol,
                              std::vector<float>& output_val) {
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
