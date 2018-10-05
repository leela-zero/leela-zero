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
OpenCLScheduler<net_t>::OpenCLScheduler() {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL<net_t>>(gpu, silent);
        auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));

        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::initialize(const int channels) {
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough threads
    // so that we can at least concurrently schedule something to the GPU.
    auto num_worker_threads = cfg_num_threads / cfg_batch_size / (m_opencl.size() + 1) + 1;
    auto gnum = 0;
    for (auto & opencl : m_opencl) {
        opencl->initialize(channels, cfg_batch_size);

        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t = std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum);
            m_worker_threads.push_back(std::move(t));
        }
        gnum++;
    }
}

template <typename net_t>
OpenCLScheduler<net_t>::~OpenCLScheduler() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto & x : m_worker_threads) {
        x.join();
    }
}

template<typename net_t>
bool OpenCLScheduler<net_t>::needs_autodetect() {
    for (auto& opencl : m_opencl) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!opencl->has_fp16_compute()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(
    unsigned int filter_size,
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

        const auto Upad = zeropad_U<net_t>(weights,
                                           outputs, channels,
                                           m_ceil, k_ceil);
        opencl_net->push_input_convolution(
            filter_size, channels, outputs,
            Upad, from_float(means), from_float(variances)
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

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto & opencl_net : m_networks) {
        opencl_net->push_convolve(filter_size, channels, outputs,
                                  from_float(weights));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_weights(
    unsigned int filter_size,
    unsigned int channels,
    unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {

    auto weight_index = size_t{0};

    // Winograd filter transformation changes filter size to 4x4
    push_input_convolution(filter_size, channels, outputs,
                           weights->m_conv_weights[weight_index],
                           weights->m_batchnorm_means[weight_index],
                           weights->m_batchnorm_stddevs[weight_index]);
    weight_index++;

    // residual blocks : except the first entry,
    // the second ~ last entry is all on residual topwer
    for (auto i = size_t{0}; i < weights->m_conv_weights.size()/2; i++) {
        push_residual(filter_size, outputs, outputs,
                      weights->m_conv_weights[weight_index],
                      weights->m_batchnorm_means[weight_index],
                      weights->m_batchnorm_stddevs[weight_index],
                      weights->m_conv_weights[weight_index + 1],
                      weights->m_batchnorm_means[weight_index + 1],
                      weights->m_batchnorm_stddevs[weight_index + 1]);
        weight_index += 2;
    }

    // Output head convolutions
    push_convolve(1, outputs, Network::OUTPUTS_POLICY, weights->m_conv_pol_w);
    push_convolve(1, outputs, Network::OUTPUTS_VALUE, weights->m_conv_val_w);
}

template <typename net_t>
void OpenCLScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val) {
    auto entry = std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.push_back(entry);
    }
    m_cv.notify_one();
    entry->cv.wait(lk);
}

#ifndef NDEBUG
std::atomic<size_t> batch_stats[2];
#endif

template <typename net_t>
void OpenCLScheduler<net_t>::batch_worker(const size_t gnum) {
    constexpr auto in_size = Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_pol_size = Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_val_size = Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE;

    OpenCLContext context;

    auto pickup_task = [this, gnum] () {
        bool is_batching = true;

        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        int count = 0;

        std::unique_lock<std::mutex> lk(m_mutex);
        while (true) {
            if (!m_running) return inputs;
            count = static_cast<int>(m_forward_queue.size());

            if (is_batching) {
                if (count < cfg_batch_size) {
                    count = 0;
                } else {
                    count = cfg_batch_size;
                }
            } else {
                if (count == 0) {
                    // empty, leave it as zero
                } else if (count < cfg_batch_size) {
                    count = 1;
                } else {
                    count = cfg_batch_size;
                }
            }

            if (count > 0) {
                auto end = begin(m_forward_queue);
                std::advance(end, count);
                std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
                m_forward_queue.erase(begin(m_forward_queue), end);

                return inputs;
            } else {
                bool timeout = !m_cv.wait_for(
                    lk,
                    std::chrono::milliseconds(m_waittime),
                    [this] () {
                        return static_cast<int>(m_forward_queue.size()) >= cfg_batch_size;
                    }
                );

                // we do this only using GPU 0.  All other GPUs will have batch capability only
                if (gnum == 0 && !m_forward_queue.empty()) {
                    if (timeout) {
                        m_waittime++;
                        is_batching = false;
                    } else if(static_cast<int>(m_forward_queue.size()) > cfg_batch_size) {
                        if(m_waittime > 1) {
                            m_waittime--;
                        }
                    }
                }
            }
        }
    };

    auto batch_input = std::vector<float>();
    auto batch_output_pol = std::vector<float>();
    auto batch_output_val = std::vector<float>();

    while (true) {
        auto inputs = pickup_task();
        auto count = inputs.size();

        if (!m_running) {
            return;
        }

#ifndef NDEBUG
        batch_stats[count == cfg_batch_size ? 1 : 0]++;
#endif

        batch_input.resize(in_size * count);
        batch_output_pol.resize(out_pol_size * count);
        batch_output_val.resize(out_val_size * count);
        {
            size_t index = 0;
            for (auto it = begin(inputs); it != end(inputs); ++it) {
                std::unique_lock<std::mutex> lk((*it)->mutex);
                std::copy(begin((*it)->in), end((*it)->in), begin(batch_input) + in_size * index);
                index++;
            }
        }

        {
            m_networks[gnum]->forward(
                batch_input, batch_output_pol, batch_output_val, context, count);
        }

        {
            size_t index = 0;
            for (auto it = begin(inputs); it != end(inputs); ++it) {
                std::copy(begin(batch_output_pol) + out_pol_size * index,
                          begin(batch_output_pol) + out_pol_size * (index + 1),
                          begin((*it)->out_p));
                std::copy(begin(batch_output_val) + out_val_size * index,
                          begin(batch_output_val) + out_val_size * (index + 1),
                          begin((*it)->out_v));
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
