/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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
#include <array>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <fstream>

#include "OpenCL.h"
#include "Tuner.h"
#include "Utils.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#define TUNER_FILE_LOCAL "leelaz_tuners"
#define MAX_ERROR 1e-4

using namespace Utils;

static void sgemmBatched_ref(const std::vector<float>& a, const std::vector<float>& b,
        std::vector<float>& c, const int m, const int n, const int k,
        const int batch_size) {

    for (auto batch = 0; batch < batch_size; batch++) {

        auto offset_u = batch*m*k;
        auto offset_v = batch*n*k;
        auto offset_m = batch*m*n;

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m, n, k,
                    1.0f,
                    &a[offset_u], m,
                    &b[offset_v], n,
                    0.0f,
                    &c[offset_m], n);
    }
}


bool IsMultiple(const size_t a, const size_t b) {
    return ((a/b)*b == a);
};

bool Tuner::valid_config_sgemm(Parameters p) {

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
    if (p["MDIMC"] != p["MDIMA"]) {
        return false;
    }
    if (p["NDIMC"] != p["NDIMB"]) {
        return false;
    }
    if (p["SA"] != p["SB"]) {
        return false;
    }
    return true;
}


Parameters Tuner::get_parameters_by_int(const std::vector<Configurations> opts, const int n) {

        Parameters param;
        std::vector<size_t> choices(opts.size());

        auto cfgs = 1;
        for (auto c = size_t{0}; c < opts.size(); c++) {
            choices[c] = opts[c].second.size();
            cfgs *= choices[c];
        }
        int j = n;

        for (auto c = size_t{0}; c < opts.size(); c++) {
            auto o = opts[c];
            auto s = o.first;
            auto v = o.second[j % choices[c]];
            j /= choices[c];
            param[s] = v;
        }

        return param;
}

std::string Tuner::parameters_to_string(const Parameters p) {
    std::string s;
    for (auto const& x : p) {
        s += " -D" + x.first + "=" + std::to_string(x.second);
    }
    return s;
}

size_t next_power_of_two(const size_t x) {
    return 2 << (size_t)(std::ceil(std::log2(x)) - 1);
}

static void sgemm_generate_data(std::vector<float> &x, const int m, const int n,
        const int batch_size, const int m_ceil, const int n_ceil) {
    for (auto batch = 0; batch < batch_size; batch++) {
        for (auto i = 0; i < n_ceil; i++) {
            if (i < n) {
                for (auto j = 0; j < m; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] = 0.01f*(((i ^ j) + batch - 50) % 100);
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

static float compare_ref(std::vector<float> &x, std::vector<float> &ref,
        const int m, const int n, const int batch_size) {

    auto sum = 0.0f;
    for (auto batch = 0; batch < batch_size; batch++) {
        for (auto i = 0; i < n; i++) {
            for (auto j = 0; j < m; j++) {
                auto r = ref[batch*n*m + i*m + j];
                auto y = x[batch*n*m + j*n + i];

                sum += (r - y) * (r - y);
            }
        }
    }
    return sum/(m*n);
}

std::string Tuner::tune_sgemm(const int m, const int n, const int k, const int batch_size, const int runs) {

    std::vector<Configurations> opts = {
      {"MWG", {16, 32, 64}},
      {"NWG", {16, 32, 64}},
      {"KWG", {32}},
      {"MDIMC", {8, 16, 32}},
      {"NDIMC", {8, 16, 32}},
      {"MDIMA", {8, 16, 32}},
      {"NDIMB", {8, 16, 32}},
      {"KWI", {2}},
      {"VWM", {1, 2, 4}},
      {"VWN", {1, 2, 4}},
      {"STRM", {0}},
      {"STRN", {0}},
      {"SA", {0, 1}},
      {"SB", {0, 1}},
    };

    size_t at_size = batch_size*next_power_of_two(k*m);
    size_t b_size = batch_size*next_power_of_two(k*n);
    size_t c_size = batch_size*next_power_of_two(m*n);

    std::vector<float> at(at_size);
    std::vector<float> b(b_size);
    std::vector<float> c(c_size);
    std::vector<float> c_ref(c_size);

    auto aBuffer = cl::Buffer(
        CL_MEM_READ_WRITE, sizeof(float)*at_size, nullptr, nullptr);
    auto bBuffer = cl::Buffer(
        CL_MEM_READ_WRITE, sizeof(float)*b_size, nullptr, nullptr);
    auto cBuffer = cl::Buffer(
        CL_MEM_READ_WRITE, sizeof(float)*c_size, nullptr, nullptr);

    myprintf("\nStarted OpenCL SGEMM tuner\n");

    std::vector<int> valid_params;

    auto cfgs = 1;
    for (auto c = size_t{0}; c < opts.size(); c++) {
        cfgs *= opts[c].second.size();
    }

    for (auto i = 0; i < cfgs; i++) {
        Parameters param = get_parameters_by_int(opts, i);
        if (valid_config_sgemm(param)) {
            valid_params.emplace_back(i);
        }
    }
    myprintf("Found %zu valid configurations\n", valid_params.size());

    std::string best_params;
    auto best_time = unsigned{0};

    auto queue = cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_PROFILING_ENABLE);

    auto event = cl::Event();

    auto program = cl::Program(sourceCode_sgemm);

    auto m_ceil_prev = 0;
    auto n_ceil_prev = 0;
    auto k_ceil_prev = 0;

    for (auto i : valid_params) {
        Parameters p = get_parameters_by_int(opts, i);
        std::string defines = parameters_to_string(p);

        try {
            std::string args = opencl.m_cl_args;

            args += defines;

            program.build(args.c_str());
        } catch (const cl::Error&) {
            //Failed to compile, get next parameter
            continue;
        }

        auto sgemm_kernel = cl::Kernel(program, "XgemmBatched");

        int m_ceil = (int)lcm(lcm(m, p["MWG"]), p["VWM"]);
        int n_ceil = (int)lcm(lcm(n, p["NWG"]), p["VWN"]);
        int k_ceil = (int)lcm(lcm(k, p["KWG"]), p["VWM"]);

        if (m_ceil != m_ceil_prev || n_ceil != n_ceil_prev || k_ceil != k_ceil_prev) {
            m_ceil_prev = m_ceil;
            n_ceil_prev = n_ceil;
            k_ceil_prev = k_ceil;

            sgemm_generate_data(at, k, m, batch_size, k_ceil, m_ceil);
            sgemm_generate_data(b, k, n, batch_size, k_ceil, n_ceil);

            sgemmBatched_ref(at, b, c_ref, m_ceil, n_ceil, k_ceil, batch_size);
            queue.enqueueWriteBuffer(aBuffer, CL_FALSE, 0, at_size*sizeof(float), at.data());
            queue.enqueueWriteBuffer(bBuffer, CL_FALSE, 0, b_size*sizeof(float), b.data());
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
                                  (size_t)batch_size};

        auto sum = 0.0;
        auto max_error = 0.0f;
        for (auto r = 0; r < runs; r++) {

            try {
                queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange,
                                            size_sgemm, local_sgemm,
                                            nullptr, &event);
                queue.finish();
                event.wait();

                queue.enqueueReadBuffer(cBuffer, CL_FALSE, 0, c_size*sizeof(float), c.data());
                queue.finish();

                max_error = std::max(max_error, compare_ref(c, c_ref, n_ceil, m_ceil, batch_size));

                auto elapsed = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                            event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

                sum += elapsed;
            } catch (const cl::Error&) {
                //Failed to enqueue kernel. Set error to max.
                max_error = MAX_ERROR;
                break;
            }
        }
        if (max_error < MAX_ERROR && (best_time == 0 || sum < best_time)) {
            myprintf("%s : %.4f ms\n", defines.c_str(), 1e-6*sum/runs);
            best_time = sum;
            best_params = defines;
        }

    }
    if (best_time == 0) {
        myprintf("Failed to find a working configuration.\nCheck your OpenCL drivers.\n");
        std::exit(-1);
    }
    return best_params;
}

void Tuner::store_sgemm_tuners(const int m, const int n, const int k, const int batch_size, std::string tuners) {
    std::string file_contents;
    {
        std::ifstream file(TUNER_FILE_LOCAL);
        if (file.good()) {
            //Read the previous contents to string
            file_contents = std::string((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        }
    }
    std::ofstream file(TUNER_FILE_LOCAL);

    //Write back previous data
    file << file_contents;

    auto device_name = opencl.get_device_name();
    std::stringstream tuning_params;
    tuning_params << m << ";" << n << ";" << k << ";" << batch_size;

    std::string line = std::to_string(TUNER_VERSION) + ";XgemmBatched;" + tuning_params.str() + ";" + tuners + ";" + device_name + "\n";

    file << line;
}


std::string Tuner::sgemm_tuners_from_line(std::string line, const int m, const int n, const int k, const int batch_size) {
    std::vector<std::string> s;

    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ';')) {
        s.emplace_back(item);
    }

    if (s.size() != 8) {
        return "";
    }

    if (s[0] != std::to_string(TUNER_VERSION)) {
        return "";
    }

    if (s[1] != "XgemmBatched") {
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

    if (s[7] != opencl.get_device_name()) {
        return "";
    }

    return s[6];
}

std::string Tuner::load_sgemm_tuners(const int m, const int n, const int k, const int batch_size) {
    std::ifstream file(TUNER_FILE_LOCAL);
    if (file.good()) {
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            auto tuners = sgemm_tuners_from_line(line, m, n, k, batch_size);
            if (tuners.size() != 0) {
                myprintf("Loaded existing SGEMM tuners\n");
                return tuners;
            }
        }
    }
    auto tuners = tune_sgemm(m, n, k, batch_size);
    store_sgemm_tuners(m, n, k, batch_size, tuners);
    return tuners;
}

#endif
