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
    auto Upad =
        std::vector<float>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
                for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

template <typename net_t>
void OpenCLScheduler<net_t>::initialize(const int channels) {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // an empty GPU list from the command line represents autodetect.
    // put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};
    auto gnum = size_t{0};

    // launch the worker thread.  round_up(cfg_num_threads / gpus.size()) threads
    // so that we only have enough contexts to achieve full parallelism.
    const auto num_threads = (cfg_num_threads + gpus.size() - 1) / gpus.size();
    m_context_pool.resize(num_threads);

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL<net_t>>();
        auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
        opencl->initialize(channels, gpu, silent);
        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));

        // starting next GPU, let's not dump full list of GPUs
        silent = true;

        for (auto i = size_t{0}; i < num_threads; i++) {
            m_context_pool[i].emplace_back(std::make_shared<ContextPoolEntry>(gnum));
        }
        gnum++;
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(unsigned int filter_size,
                                                    unsigned int channels,
                                                    unsigned int outputs,
                                                    const std::vector<float>& weights,
                                                    const std::vector<float>& means,
                                                    const std::vector<float>& variances) {
    for (const auto& opencl_net : m_networks) {
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

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights_1,
                                           const std::vector<float>& means_1,
                                           const std::vector<float>& variances_1,
                                           const std::vector<float>& weights_2,
                                           const std::vector<float>& means_2,
                                           const std::vector<float>& variances_2) {
    for (const auto& opencl_net : m_networks) {
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

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto & opencl_net : m_networks) {
        opencl_net->push_convolve(filter_size, channels, outputs, weights);
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val) {
    std::shared_ptr<ContextPoolEntry> ctx;
    auto queue_num = size_t{0};
    {
        LOCK(m_context_pool_mutex, lock);
        while (queue_num < m_context_pool.size()) {
            if (!m_context_pool[queue_num].empty()) {
                ctx = std::move(m_context_pool[queue_num].front());
                m_context_pool[queue_num].pop_front();
                break;
            }
            queue_num++;
        }
        // if this failed, it means we ran out of contexts
        // which should be more than or equal to the number of threads
        assert(ctx != nullptr);
    }

    m_networks[ctx->net_index]->forward(input, output_pol, output_val, ctx->context);

    {
        LOCK(m_context_pool_mutex, lock);
        m_context_pool[queue_num].push_back(std::move(ctx));
    }
}

template class OpenCLScheduler<float>;
#ifdef USE_HALF
template class OpenCLScheduler<half_float::half>;
#endif

#endif
