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
#include "config.h"

#ifdef USE_OPENCL

#include "GTP.h"
#include "Network.h"
#include "OpenCLScheduler.h"
#include "Random.h"
#include "Utils.h"

using Utils::ceilMultiple;
using Utils::myprintf;

class from_float {
public:
    from_float(const std::vector<float>& f) : m_f(f) {}

    operator const std::vector<float> &() {
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
static std::vector<T> zeropad_U(const std::vector<float>& U, const int outputs,
                                const int channels, const int outputs_pad,
                                const int channels_pad) {
    // Fill with zeroes
    auto Upad = std::vector<T>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            for (auto c = 0; c < channels; c++) {
                for (auto o = 0; o < outputs; o++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad) + c * outputs_pad
                         + o] =
                        U[xi * (WINOGRAD_ALPHA * outputs * channels)
                          + nu * (outputs * channels) + c * outputs + o];
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
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough
    // threads so that we can at least concurrently schedule something to the
    // GPU.
    auto num_worker_threads =
        cfg_num_threads / cfg_batch_size / (m_opencl.size() + 1) + 1;
    auto gnum = 0;
    for (auto& opencl : m_opencl) {
        opencl->initialize(channels, cfg_batch_size);

        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t =
                std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum);
            m_worker_threads.push_back(std::move(t));
        }
        gnum++;
    }

    // Exit immediately after tuning.  We should exit here because we skipped
    // initializing rest of the kernels due to some NVIDIA drivers crashing.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

template <typename net_t>
OpenCLScheduler<net_t>::~OpenCLScheduler() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto& x : m_worker_threads) {
        x.join();
    }
}

template <typename net_t>
bool OpenCLScheduler<net_t>::needs_autodetect() {
    for (auto& opencl : m_opencl) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!opencl->has_fp16_compute() && !opencl->has_tensor_cores()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(
    const unsigned int filter_size, const unsigned int channels,
    const unsigned int outputs,
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

        const auto Upad =
            zeropad_U<net_t>(weights, outputs, channels, m_ceil, k_ceil);
        opencl_net->push_input_convolution(filter_size, channels, outputs, Upad,
                                           from_float(means),
                                           from_float(variances));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual(
    const unsigned int filter_size, const unsigned int channels,
    const unsigned int outputs,
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
        const auto Upad1 =
            zeropad_U<net_t>(weights_1, outputs, outputs, m_ceil, m_ceil);
        const auto Upad2 =
            zeropad_U<net_t>(weights_2, outputs, outputs, m_ceil, m_ceil);
        opencl_net->push_residual(filter_size, channels, outputs,
                                  Upad1, from_float(means_1),
                                  from_float(variances_1),
                                  Upad2, from_float(means_2),
                                  from_float(variances_2));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_se(const unsigned int channels,
                                     const unsigned int outputs,
                                     const std::vector<float>& se_fc1_w,
                                     const std::vector<float>& se_fc1_b,
                                     const std::vector<float>& se_fc2_w,
                                     const std::vector<float>& se_fc2_b) {
    for (const auto& opencl_net : m_networks) {
        opencl_net->push_se(channels, outputs,
                            from_float(se_fc1_w), from_float(se_fc1_b),
                            from_float(se_fc2_w), from_float(se_fc2_b));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(const unsigned int filter_size,
                                           const unsigned int channels,
                                           const unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto& opencl_net : m_networks) {
        opencl_net->push_convolve(filter_size, channels, outputs,
                                  from_float(weights));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_weights(
    const unsigned int filter_size, const unsigned int channels,
    const unsigned int outputs,
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
    for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
        push_residual(filter_size, outputs, outputs,
                      weights->m_conv_weights[weight_index],
                      weights->m_batchnorm_means[weight_index],
                      weights->m_batchnorm_stddevs[weight_index],
                      weights->m_conv_weights[weight_index + 1],
                      weights->m_batchnorm_means[weight_index + 1],
                      weights->m_batchnorm_stddevs[weight_index + 1]);
        if (weights->m_se_fc1_w.size() > 0) {
            const auto se_fc_outputs = weights->m_se_fc1_b[0].size();
            push_se(outputs, se_fc_outputs,
                    weights->m_se_fc1_w[i], weights->m_se_fc1_b[i],
                    weights->m_se_fc2_w[i], weights->m_se_fc2_b[i]);
        }
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
    auto entry =
        std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.push_back(entry);

        if (m_single_eval_in_progress.load()) {
            m_waittime += 2;
        }
    }
    m_cv.notify_one();
    entry->cv.wait(lk);

    if (m_draining) {
        throw NetworkHaltException();
    }
}

#ifndef NDEBUG
struct batch_stats_t batch_stats;
#endif

template <typename net_t>
void OpenCLScheduler<net_t>::batch_worker(const size_t gnum) {
    constexpr auto in_size = Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_pol_size =
        Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_val_size =
        Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE;

    OpenCLContext context;

    // batch scheduling heuristic.
    // Returns the batch picked up from the queue (m_forward_queue)
    // 1) Wait for m_waittime milliseconds for full batch
    // 2) if we don't have a full batch then just do a single eval
    //
    // The purpose of m_waittime is to prevent the system from deadlocking
    // because we were waiting for a job too long, while the job is never
    // going to come due to a control dependency (e.g., evals stuck on a
    // critical path).  To do so:
    //
    // 1) if we couldn't form a batch after waiting m_waittime ms, it means
    // that we hit the critical path and should do scalar evals.
    // Wait 1ms shorter next time.
    //
    // 2) if we picked up a single eval, but were getting additional evals
    // while that single eval was being processed, it means that we made
    // the wrong decision.  Wait 2ms longer next time.

    auto pickup_task = [this]() {
        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        size_t count = 0;

        std::unique_lock<std::mutex> lk(m_mutex);
        while (true) {
            if (!m_running) {
                return inputs;
            }
            count = m_forward_queue.size();
            if (count >= cfg_batch_size) {
                count = cfg_batch_size;
                break;
            }

            bool timeout = !m_cv.wait_for(
                lk, std::chrono::milliseconds(m_waittime), [this]() {
                    return !m_running
                           || m_forward_queue.size() >= cfg_batch_size;
                });

            if (!m_forward_queue.empty()) {
                if (timeout
                    && m_single_eval_in_progress.exchange(true) == false) {
                    // Waited long enough but couldn't form a batch.
                    // Check if there is any other single eval in progress,
                    // and if not, do one from this thread.
                    if (m_waittime > 1) {
                        m_waittime--;
                    }
                    count = 1;
                    break;
                }
            }
        }
        // Move 'count' evals from shared queue to local list.
        auto end = begin(m_forward_queue);
        std::advance(end, count);
        std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(begin(m_forward_queue), end);

        return inputs;
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
        if (count == 1) {
            batch_stats.single_evals++;
        } else {
            batch_stats.batch_evals++;
        }
#endif

        // prepare input for forward() call
        batch_input.resize(in_size * count);
        batch_output_pol.resize(out_pol_size * count);
        batch_output_val.resize(out_val_size * count);

        auto index = size_t{0};
        for (auto& x : inputs) {
            std::unique_lock<std::mutex> lk(x->mutex);
            std::copy(begin(x->in), end(x->in),
                      begin(batch_input) + in_size * index);
            index++;
        }

        // run the NN evaluation
        m_networks[gnum]->forward(batch_input, batch_output_pol,
                                  batch_output_val, context, count);

        // Get output and copy back
        index = 0;
        for (auto& x : inputs) {
            std::copy(begin(batch_output_pol) + out_pol_size * index,
                      begin(batch_output_pol) + out_pol_size * (index + 1),
                      begin(x->out_p));
            std::copy(begin(batch_output_val) + out_val_size * index,
                      begin(batch_output_val) + out_val_size * (index + 1),
                      begin(x->out_v));
            x->cv.notify_all();
            index++;
        }

        if (count == 1) {
            m_single_eval_in_progress = false;
        }
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::drain() {
    // When signaled to drain requests, this method picks up all pending
    // requests and wakes them up.  Throws exception once the woken up request
    // sees m_draining.
    m_draining = true;

    std::list<std::shared_ptr<ForwardQueueEntry>> fq;
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        std::move(m_forward_queue.begin(), m_forward_queue.end(),
                  std::back_inserter(fq));
        m_forward_queue.clear();
    }

    for (auto& x : fq) {
        {
            // dummy lock/unlock to make sure thread in forward() is sleeping
            std::unique_lock<std::mutex> lk(x->mutex);
        }
        x->cv.notify_all();
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::resume() {
    // UCTNode::think() should wait for all child threads to complete before resuming.
    assert(m_forward_queue.empty());

    m_draining = false;
}

template class OpenCLScheduler<float>;
#ifdef USE_HALF
template class OpenCLScheduler<half_float::half>;
#endif

#endif
