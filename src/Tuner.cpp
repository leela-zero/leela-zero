/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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
#include <array>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <fstream>
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#include "GTP.h"
#include "OpenCL.h"
#include "Tuner.h"
#include "Utils.h"
#include "Random.h"

const auto TUNER_FILE_LOCAL = std::string("leelaz_opencl_tuning");

template <typename net_t>
std::vector<std::string> Tuner<net_t>::tuned_devices;

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

template <typename net_t> static std::string getTunerKernel();
template <typename net_t> static float getTunerMaxError();

template <> std::string getTunerKernel<float>() {
    return std::string("XgemmBatched");
}

template <> float getTunerMaxError<float>() {
    return 1e-4f;
}

#ifdef USE_HALF
template <> std::string getTunerKernel<half_float::half>() {
    return std::string("XgemmBatchedHalf");
}

template <> float getTunerMaxError<half_float::half>() {
    return 1e-1f;
}
#endif

using namespace Utils;

template <typename net_t>
static void sgemmBatched_ref(const std::vector<net_t>& a,
                             const std::vector<net_t>& b,
                             std::vector<net_t>& c,
                             const int m, const int n, const int k,
                             const int batch_size) {
    std::vector<float> ar(a.size());
    std::vector<float> br(b.size());
    std::vector<float> cr(c.size());

    std::copy(begin(a), end(a), begin(ar));
    std::copy(begin(b), end(b), begin(br));

    for (auto batch = 0; batch < batch_size; batch++) {
        auto offset_u = batch * m * k;
        auto offset_v = batch * n * k;
        auto offset_m = batch * m * n;
#ifdef USE_BLAS
        // Calculates C = transpose(tranpose(A) * B) in row major, or
        // C = A * transpose(B) in column major.
        for (auto i = 0; i < m; i++) {
            for (auto j = 0; j < n; j++) {
                auto acc = 0.0f;
                for (auto l = 0; l < k; l++) {
                    acc += ar[l * m + i + offset_u] * br[l * n + j + offset_v];
                }
                cr[j * m + i + offset_m] = acc;
            }
        }
#else
        auto C = EigenMatrixMap<float>(cr.data() + offset_m, m, n);
        auto A = ConstEigenMatrixMap<float>(ar.data() + offset_u, m, k);
        auto B = ConstEigenMatrixMap<float>(br.data() + offset_v, n, k);
        C.noalias() = (A * B.transpose());
#endif
    }

    std::copy(begin(cr), end(cr), begin(c));
}


static bool IsMultiple(const size_t a, const size_t b) {
    return (a % b == 0);
}

template <typename net_t>
bool Tuner<net_t>::valid_config_sgemm(Parameters p, bool exhaustive) {
    if (p["TCE"] == 0) {
        if (!IsMultiple(p["MWG"], p["MDIMC"]*p["VWM"])) {
            return false;
        }
        if (!IsMultiple(p["NWG"], p["NDIMC"]*p["VWN"])) {
            return false;
        }
        if (!IsMultiple(p["MWG"], p["MDIMA"]*p["VWM"])) {
            return false;
        }
        if (!IsMultiple(p["NWG"], p["NDIMB"]*p["VWN"])) {
            return false;
        }
        if (!IsMultiple(p["KWG"], p["MDIMC"]*p["NDIMC"]/p["MDIMA"])) {
            return false;
        }
        if (!IsMultiple(p["KWG"], p["MDIMC"]*p["NDIMC"]/p["NDIMB"])) {
            return false;
        }
        // Extra restrictions for a fast tuning run
        if (!exhaustive) {
            if (p["MDIMC"] != p["MDIMA"]) {
                return false;
            }
            if (p["NDIMC"] != p["NDIMB"]) {
                return false;
            }
            if (p["SA"] != p["SB"]) {
                return false;
            }
        }
    } else {
        if (!m_use_tensorcore) {
            return false;
        }

        // In Tensor Core implementations, MDIMA and NDIMB represents the
        // wmmv multiplication dimensions, that is,
        // m16n16k16 / m32n8k16 / m8n32k16.  Thus m * n is fixed to 256.
        if (p["MDIMA"] * p["NDIMB"] != 256) {
            return false;
        }
        if (p["MWG"] < p["MDIMC"]) {
            return false;
        }
        if (p["NWG"] < p["NDIMC"]) {
            return false;
        }
        if (p["MDIMC"] < p["MDIMA"]) {
            return false;
        }
        if (p["NDIMC"] < p["NDIMB"]) {
            return false;
        }
        // VWM / VWN has no meaning if we don't do SA / SB.
        // Only test VWM / VWN == 2
        if (p["SA"] == 0 && p["VWM"] != 2) {
            return false;
        }
        if (p["SB"] == 0 && p["VWN"] != 2) {
            return false;
        }
    }
    return true;
}

template <typename net_t>
Parameters Tuner<net_t>::get_parameters_by_int(
    const std::vector<Configurations>& opts, const int n) {

    Parameters param;
    std::vector<size_t> choices(opts.size());

    auto cfgs = 1;
    for (auto c = size_t{0}; c < opts.size(); c++) {
        choices[c] = opts[c].second.size();
        cfgs *= choices[c];
    }
    auto j = n;

    for (auto c = size_t{0}; c < opts.size(); c++) {
        auto o = opts[c];
        auto s = o.first;
        auto v = o.second[j % choices[c]];
        j /= choices[c];
        param[s] = v;
    }

    return param;
}

template <typename net_t>
std::string Tuner<net_t>::parameters_to_defines(const Parameters& p) {
    std::string s;
    for (auto const& x : p) {
        s += " -D" + x.first + "=" + std::to_string(x.second);
    }
    return s;
}

template <typename net_t>
std::string Tuner<net_t>::parameters_to_string(const Parameters& p) {
    std::string s;
    for (auto const& x : p) {
        s += x.first + "=" + std::to_string(x.second) + " ";
    }
    if (s.size() > 0) {
        s.resize(s.size() - 1);
    }
    return s;
}

static size_t next_power_of_two(const size_t x) {
    return 2 << size_t(std::ceil(std::log2(x)) - 1);
}

template <typename net_t>
static void sgemm_generate_data(std::vector<net_t> &x,
                                const int m, const int n,
                                const int batch_size,
                                const int m_ceil, const int n_ceil) {
    for (auto batch = 0; batch < batch_size; batch++) {
        for (auto i = 0; i < n_ceil; i++) {
            if (i < n) {
                for (auto j = 0; j < m; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] =
                        (( (i ^ j) + batch - 128) % 256) / 256.0f;
                }
                for (auto j = m; j < m_ceil; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] = 0.0f;
                }
            } else {
                for (auto j = 0; j < m_ceil; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] = 0.0f;
                }
            }
        }
    }
}

template <typename net_t>
static float compare_ref(std::vector<net_t> &x, std::vector<net_t> &ref,
                         const int m, const int n, const int batch_size,
                         const int m_ceil, const int n_ceil) {
    auto sum = 0.0f;
    for (auto batch = 0; batch < batch_size; batch++) {
        for (auto j = 0; j < m; j++) {
            for (auto i = 0; i < n; i++) {
                auto r = ref[batch*n*m + j*n + i];
                auto y = x[batch*n_ceil*m_ceil + j*n_ceil + i];

                sum += (r - y) * (r - y);
            }
        }
    }
    return sum / (m * n * batch_size);
}

template <typename net_t>
std::vector<Parameters> Tuner<net_t>::build_valid_params() {
    auto opts = std::vector<Configurations>();
    if (cfg_sgemm_exhaustive) {
        opts = {
            {"MWG", {16, 32, 64}},
            {"NWG", {16, 32, 64}},
            {"KWG", {16, 32}},
            {"MDIMC", {8, 16, 32}},
            {"NDIMC", {8, 16, 32}},
            {"MDIMA", {8, 16, 32}},
            {"NDIMB", {8, 16, 32}},
            {"KWI", {2, 8}},
            {"VWM", {1, 2, 4, 8}},
            {"VWN", {1, 2, 4, 8}},
            {"STRM", {0, 1}},
            {"STRN", {0, 1}},
            {"SA", {0, 1}},
            {"SB", {0, 1}},
        };
    } else {
        opts = {
            {"MWG", {16, 32, 64}},
            {"NWG", {16, 32, 64}},
            {"KWG", {16, 32}},
            {"MDIMC", {8, 16, 32}},
            {"NDIMC", {8, 16, 32}},
            {"MDIMA", {8, 16, 32}},
            {"NDIMB", {8, 16, 32}},
            {"KWI", {2, 8}},
            {"VWM", {2, 4}},
            {"VWN", {2, 4}},
            {"STRM", {0}},
            {"STRN", {0}},
            {"SA", {1}},
            {"SB", {1}},
        };
    }
    // Tensor Core options
    auto topts = std::vector<Configurations>();
    if (cfg_sgemm_exhaustive) {
        topts = {
            {"MWG", {32, 64, 128, 256}},
            {"NWG", {8, 16, 32, 64}},
            {"KWG", {16, 32, 64}},
            {"MDIMC", {8, 16, 32, 64}},
            {"NDIMC", {8, 16, 32, 64}},
            {"MDIMA", {8, 16, 32}},
            {"NDIMB", {8, 16, 32}},
            {"KWI", {2}},
            {"VWM", {2, 4, 8}},
            {"VWN", {2, 4, 8}},
            {"STRM", {0}},
            {"STRN", {0}},
            {"SA", {0, 1}},
            {"SB", {0, 1}},
        };
    } else {
        topts = {
            {"MWG", {32, 64, 128}},
            {"NWG", {8, 16, 32}},
            {"KWG", {16, 32}},
            {"MDIMC", {8, 16, 32}},
            {"NDIMC", {8, 16, 32}},
            {"MDIMA", {8, 16, 32}},
            {"NDIMB", {8, 16, 32}},
            {"KWI", {2}},
            {"VWM", {2}},
            {"VWN", {2}},
            {"STRM", {0}},
            {"STRN", {0}},
            {"SA", {0}},
            {"SB", {0}},
        };
    }

    auto valid_params = std::vector<Parameters>{};
    auto build_from = [this, &valid_params](std::vector<Configurations> & opts, int tce) {
        auto cfgs = 1;
        for (auto c = size_t{0}; c < opts.size(); c++) {
            cfgs *= opts[c].second.size();
        }
        for (auto i = 0; i < cfgs; i++) {
            Parameters param = get_parameters_by_int(opts, i);
            param["TCE"] = tce;
            if (valid_config_sgemm(param, cfg_sgemm_exhaustive)) {
                valid_params.push_back(param);
            }
        }
    };
    build_from(opts, 0);
    build_from(topts, 1);

    // Don't use thread RNG or determinism will depend on whether tuner ran.
    auto rng = Random{0};
    std::shuffle(begin(valid_params), end(valid_params), rng);

    if (cfg_sgemm_exhaustive) {
        // Likely too many valid params, cut out some of them
        valid_params.resize(valid_params.size() / 16);
    }

    return valid_params;
}

template <typename net_t>
std::string Tuner<net_t>::tune_sgemm(const int m, const int n, const int k,
                              const int batch_size, const int runs) {
    // This needs to be at minimum the maximum (MNK/WG) values above.
    auto m_max = std::max(64, m);
    auto n_max = std::max(64, n);
    auto k_max = std::max(32, k);

    auto at_size = batch_size
        * next_power_of_two(k_max) * next_power_of_two(m_max);
    auto b_size = batch_size
        * next_power_of_two(k_max) * next_power_of_two(n_max);
    auto c_size = batch_size
        * next_power_of_two(m_max) * next_power_of_two(n_max);

    auto total_flops = batch_size * 2.0 * m * n * k;

    auto at = std::vector<net_t>(at_size);
    auto b = std::vector<net_t>(b_size);
    auto c = std::vector<net_t>(c_size);
    auto c_ref = std::vector<net_t>(c_size);

    sgemm_generate_data(at, k, m, batch_size, k, m);
    sgemm_generate_data(b, n, k, batch_size, n, k);

    sgemmBatched_ref(at, b, c_ref, m, n, k, batch_size);

    auto aBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_WRITE, sizeof(net_t) * at_size, nullptr, nullptr);
    auto bBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_WRITE, sizeof(net_t) * b_size, nullptr, nullptr);
    auto cBuffer = cl::Buffer(
        m_context,
        CL_MEM_READ_WRITE, sizeof(net_t) * c_size, nullptr, nullptr);

    myprintf("\nStarted OpenCL SGEMM tuner.\n");

    auto valid_params = build_valid_params();

    myprintf("Will try %zu valid configurations.\n", valid_params.size());

    std::string best_params;
    auto best_time = unsigned{0};

    auto queue = cl::CommandQueue(m_context,
                                  m_device,
                                  CL_QUEUE_PROFILING_ENABLE);
    auto event = cl::Event();
    auto program = cl::Program(m_context, sourceCode_common + sourceCode_sgemm);

    auto m_ceil_prev = 0;
    auto n_ceil_prev = 0;
    auto k_ceil_prev = 0;
    auto param_counter = size_t{0};
    auto min_error = 100.0f;
    auto failed_compile = 0;
    auto failed_enqueue = 0;
    auto failed_error = 0;

    for (auto & p : valid_params) {
        param_counter++;

        auto defines = parameters_to_defines(p);

        try {
            auto args = m_opencl.m_cl_args + " " + defines;
            program.build(args.c_str());
        } catch (const cl::Error&) {
            // Failed to compile, get next parameter
            failed_compile++;
            continue;
        }

        auto sgemm_kernel = cl::Kernel(program, "XgemmBatched");

        auto m_ceil = int(ceilMultiple(ceilMultiple(m, p["MWG"]), p["VWM"]));
        auto n_ceil = int(ceilMultiple(ceilMultiple(n, p["NWG"]), p["VWN"]));
        auto k_ceil = int(ceilMultiple(ceilMultiple(k, p["KWG"]), p["VWM"]));

        if (m_ceil != m_ceil_prev
            || n_ceil != n_ceil_prev
            || k_ceil != k_ceil_prev) {
            m_ceil_prev = m_ceil;
            n_ceil_prev = n_ceil;
            k_ceil_prev = k_ceil;

            sgemm_generate_data(at, k, m, batch_size, k_ceil, m_ceil);
            sgemm_generate_data(b, n, k, batch_size, n_ceil, k_ceil);

            queue.enqueueWriteBuffer(aBuffer, CL_FALSE, 0,
                                     at_size * sizeof(net_t), at.data());
            queue.enqueueWriteBuffer(bBuffer, CL_FALSE, 0,
                                     b_size * sizeof(net_t), b.data());
            queue.finish();
        }

        sgemm_kernel.setArg(0, m_ceil);
        sgemm_kernel.setArg(1, n_ceil);
        sgemm_kernel.setArg(2, k_ceil);
        sgemm_kernel.setArg(3, aBuffer);
        sgemm_kernel.setArg(4, bBuffer);
        sgemm_kernel.setArg(5, cBuffer);

        cl::NDRange local_sgemm = {p["MDIMC"], p["NDIMC"], 1};


        cl::NDRange size_sgemm = {(m_ceil * p["MDIMC"]) / p["MWG"],
                                  (n_ceil * p["NDIMC"]) / p["NWG"],
                                  size_t(batch_size)};
        // Tensor Core implementation uses a different dimension.
        if (p["TCE"]) {
            local_sgemm = {32 * p["MDIMC"] / p["MDIMA"], p["NDIMC"] / p["NDIMB"], 1};
            size_sgemm = {32 * m_ceil / p["MDIMA"] * p["MDIMC"] / p["MWG"],
                          n_ceil / p["NDIMB"] * p["NDIMC"] / p["NWG"],
                          size_t(batch_size)};
        }

        auto sum = 0.0f;
        auto error = 0.0f;

        for (auto r = 0; r < runs; r++) {
            try {
                queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange,
                                           size_sgemm, local_sgemm,
                                           nullptr, &event);
                queue.finish();
                event.wait();

                queue.enqueueReadBuffer(cBuffer, CL_FALSE, 0,
                                        c_size * sizeof(net_t), c.data());
                queue.finish();

                auto this_error = compare_ref(c, c_ref, n, m, batch_size,
                                              n_ceil, m_ceil);
                error = std::max(error, this_error);

                auto elapsed =
                    event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

                sum += elapsed;
            } catch (const cl::Error&) {
                // Failed to enqueue kernel. Set error to some big number.
                failed_enqueue++;
                error = std::numeric_limits<float>::max();
                // This failure will be counted to be failed due to error,
                // so preemptively subtract one from that count.
                failed_error--;
                break;
            }
        }

        min_error = std::min(min_error, error);

        if (error >= getTunerMaxError<net_t>()) {
            failed_error++;
        }

        if (error < getTunerMaxError<net_t>() && (best_time == 0 || sum < best_time)) {
            auto param_str = parameters_to_string(p);
            auto kernel_ms = 1e-6f * (sum / runs);
            // Timing is in nanoseconds (10^-9), Giga = 10^9, so this works out
            auto kernel_gflops = total_flops / (sum / runs);
            myprintf("(%u/%u) %s %.4f ms (%.1f GFLOPS)\n",
               param_counter, valid_params.size(), param_str.c_str(),
               kernel_ms, kernel_gflops);
            best_time = sum;
            best_params = defines;
        }
    }
    if (best_time == 0) {
        if (failed_compile > 0) {
            myprintf_error("Failed to compile: %d kernels.\n", failed_compile);
        }
        if (failed_enqueue > 0) {
            myprintf_error("Failed to enqueue: %d kernels\n", failed_enqueue);
        }
        if (failed_error > 0) {
            myprintf_error("Too high error: %d kernels\n", failed_error);
        }
        myprintf_error("Failed to find a working configuration.\nCheck your OpenCL drivers.\n");
        myprintf_error("Minimum error: %f. Error bound: %f\n", min_error, getTunerMaxError<net_t>());
        throw std::runtime_error("Tuner failed to find working configuration.");
    }
    return best_params;
}

template <typename net_t>
void Tuner<net_t>::store_sgemm_tuners(const int m, const int n, const int k,
                               const int batch_size, std::string tuners) {
    auto tuner_file = leelaz_file(TUNER_FILE_LOCAL);
    auto file_contents = std::vector<std::string>();
    {
        // Read the previous contents to string
        auto file = std::ifstream{tuner_file};
        if (file.good()) {
            auto line = std::string{};
            while (std::getline(file, line)) {
                file_contents.emplace_back(line);
            }
        }
    }
    auto file = std::ofstream{tuner_file};

    auto device_name = m_opencl.get_device_name();
    auto tuning_params = std::stringstream{};
    tuning_params << m << ";" << n << ";" << k << ";" << batch_size;

    auto tuning_line_prefix = std::to_string(TUNER_VERSION) + ";"
        + getTunerKernel<net_t>() + ";" + tuning_params.str() + ";";
    auto tuning_line = tuning_line_prefix + tuners + ";" + device_name;

    // Write back previous data as long as it's not the device and
    // tuning we just tuned
    for (const auto& line : file_contents) {
        if (line.find(tuning_line_prefix) == std::string::npos
            || line.find(device_name) == std::string::npos) {
            file << line << std::endl;
        }
    }

    // Write new tuning
    file << tuning_line << std::endl;

    if (file.fail()) {
        myprintf("Could not save the tuning result.\n");
        myprintf("Do I have write permissions on %s?\n",
            tuner_file.c_str());
    }
}

template <typename net_t>
std::string Tuner<net_t>::sgemm_tuners_from_line(std::string line,
                                          const int m, const int n, const int k,
                                          const int batch_size) {
    auto s = std::vector<std::string>{};
    auto ss = std::stringstream{line};
    auto item = std::string{};

    while (std::getline(ss, item, ';')) {
        s.emplace_back(item);
    }

    if (s.size() != 8) {
        return "";
    }

    if (s[0] != std::to_string(TUNER_VERSION)) {
        return "";
    }

    if (s[1] != getTunerKernel<net_t>()) {
        return "";
    }

    if (s[2] != std::to_string(m)) {
        return "";
    }

    if (s[3] != std::to_string(n)) {
        return "";
    }

    if (s[4] != std::to_string(k)) {
        return "";
    }

    if (s[5] != std::to_string(batch_size)) {
        return "";
    }

    if (s[7] != m_opencl.get_device_name()) {
        return "";
    }

    return s[6];
}

template <typename net_t>
std::string Tuner<net_t>::load_sgemm_tuners(const int m, const int n, const int k,
                                     const int batch_size) {
    auto tuner_file = leelaz_file(TUNER_FILE_LOCAL);
    auto file = std::ifstream{tuner_file};

    auto try_prior_tuning = file.good();

    // If we want full tuning, don't reuse previously tuned results
    // except if the tuning was created from this run from a different
    // GPU instance with the same name.  This prevents the tuner running
    // for multiple times if the system has multiple same GPUs.
    if (try_prior_tuning && cfg_sgemm_exhaustive) {
        auto dev = m_opencl.get_device_name();
        try_prior_tuning = std::any_of(
            begin(tuned_devices),
            end(tuned_devices),
            [&dev](const std::string & x) { return dev == x; }
        );
    }
    tuned_devices.emplace_back(m_opencl.get_device_name());

    if (try_prior_tuning) {
        auto line = std::string{};
        while (std::getline(file, line)) {
            auto tuners = sgemm_tuners_from_line(line, m, n, k, batch_size);
            if (tuners.size() != 0) {
                myprintf("Loaded existing SGEMM tuning.\n");
                return tuners;
            }
        }
    }
    auto tuners = tune_sgemm(m, n, k, batch_size);
    store_sgemm_tuners(m, n, k, batch_size, tuners);
    return tuners;
}

template <typename net_t>
void Tuner<net_t>::enable_tensorcore() {}

template <>
void Tuner<half_float::half>::enable_tensorcore() {
    m_use_tensorcore = true;
}

template class Tuner<float>;
#ifdef USE_HALF
template class Tuner<half_float::half>;
#endif

#endif
