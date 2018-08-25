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
using Utils::myprintf;

class from_float{
public:
    from_float(const std::vector<float> & f) : m_f(f) {}

    operator const std::vector<float>&() {
        return m_f;
    }

    operator std::vector<half_float::half>() {
        auto ret = std::vector<half_float::half>(m_f.size());
        std::copy(cbegin(m_f), cend(m_f), begin(ret));
        return ret;
    }
private:
    const std::vector<float>& m_f;
};

template <typename T>
static std::vector<T> zeropad_U(const std::vector<float>& U,
                                const int outputs, const int channels,
                                const int outputs_pad,
                                const int channels_pad) {
    // Fill with zeroes
    auto Upad =
        std::vector<T>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            for (auto c = 0; c < channels; c++) {
                for (auto o = 0; o < outputs; o++) {
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
        m_opencl.emplace_back();
        m_networks.emplace_back();
        for (int i=0; i<cfg_batch_size; i++) {
            auto opencl = std::make_unique<OpenCL<net_t>>();
            auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
            opencl->initialize(channels, gpu, silent, i+1);
            m_opencl[gnum].push_back(std::move(opencl));
            m_networks[gnum].push_back(std::move(net));
        }

        // starting next GPU, let's not dump full list of GPUs
        silent = true;

        for (auto i = size_t{0}; i < num_threads; i++) {
            m_context_pool[i].emplace_back(std::make_shared<ContextPoolEntry>(gnum));
        }

        auto t = std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum);
        t.detach();
        t = std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum);
        t.detach();
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
    for (const auto& net2 : m_networks) {
        for (const auto& opencl_net : net2) {
            const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
    
            const auto mwg = tuners[0];
            const auto kwg = tuners[2];
            const auto vwm = tuners[3];
    
            const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
            const auto k_ceil = ceilMultiple(ceilMultiple(channels, kwg), vwm);
    
            const auto Upad = zeropad_U<net_t>(weights,
                                               outputs, channels,
                                               m_ceil, k_ceil);
            opencl_net->push_input_convolution(
                filter_size, channels, outputs,
                Upad, from_float(means), from_float(variances)
            );
        }
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
    for (const auto& net2 : m_networks) {
        for (const auto& opencl_net : net2) {
            const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
    
            const auto mwg = tuners[0];
            const auto vwm = tuners[3];
    
            const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
            const auto Upad1 = zeropad_U<net_t>(weights_1,
                                                outputs, outputs,
                                                m_ceil, m_ceil);
            const auto Upad2 = zeropad_U<net_t>(weights_2,
                                                outputs, outputs,
                                                m_ceil, m_ceil);
            opencl_net->push_residual(filter_size, channels, outputs,
                                      Upad1,
                                      from_float(means_1),
                                      from_float(variances_1),
                                      Upad2,
                                      from_float(means_2),
                                      from_float(variances_2));
        }
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto & net2 : m_networks) {
        for (const auto & opencl_net : net2) {
            opencl_net->push_convolve(filter_size, channels, outputs, from_float(weights));
        }
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val) {
    auto entry = std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
#ifdef USE_LOCK_FREE_QUEUE
    m_forward_queue.enqueue(entry);
#else
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.push_back(entry);
    }
    m_cv.notify_one();
#endif
    std::unique_lock<std::mutex> lk(entry->mutex);
    entry->cv.wait(lk);
}

std::atomic<size_t> batch_index;
std::atomic<size_t> batch_stats[MAX_BATCH+1];

template <typename net_t>
void OpenCLScheduler<net_t>::batch_worker(const size_t gnum) {
    std::vector<OpenCLContext> contexts(cfg_batch_size);
    myprintf("worker %d started, batch size %d\n", gnum, cfg_batch_size);
    while (true) {
        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        size_t count = 0;

        {
#ifdef USE_LOCK_FREE_QUEUE
            count = m_forward_queue.try_dequeue_bulk(std::inserter(inputs, inputs.begin()), cfg_batch_size);
            if (count == 0) {
                continue;
            }
#else
            std::unique_lock<std::mutex> lk(m_mutex);
            count = std::min(m_forward_queue.size(), size_t(cfg_batch_size));
            if (count > 0) {
                auto begin = m_forward_queue.begin();
                auto end = begin;
                std::advance(end, count);
                std::move(begin, end, std::back_inserter(inputs));
                m_forward_queue.erase(begin, end);
            }
            else {
                m_cv.wait(lk);
                continue;
            }
#endif
        }

        auto & context = contexts[count-1];
        auto batch_input = std::vector<float>(Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE * count);
        auto batch_output_pol = std::vector<float>(Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE * count);
        auto batch_output_val = std::vector<float>(Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE * count);

        batch_index++;
        batch_stats[count]++;

        {
            size_t index = 0;
            for (auto it = inputs.begin(); it != inputs.end(); ++it) {
                std::copy((*it)->in.begin(), (*it)->in.end(), batch_input.begin() + Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE * index);
                index++;
            }
        }

        {
            m_networks[gnum][count-1]->forward(
                batch_input, batch_output_pol, batch_output_val, context, count);
        }

        {
            size_t index = 0;
            for (auto it = inputs.begin(); it != inputs.end(); ++it) {
                std::copy(batch_output_pol.begin() + Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE * index,
                          batch_output_pol.begin() + Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE * (index + 1),
                          (*it)->out_p.begin());
                std::copy(batch_output_val.begin() + Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE * index,
                          batch_output_val.begin() + Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE * (index + 1),
                          (*it)->out_v.begin());
                (*it)->cv.notify_all();
                index++;
            }
        }
    }
}

template class OpenCLScheduler<float>;
#ifdef USE_HALF
template class OpenCLScheduler<half_float::half>;
#endif

#endif
