/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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

#ifndef SGEMM_TUNER_H_INCLUDED
#define SGEMM_TUNER_H_INCLUDED

#include "config.h"
#include <vector>
#include <map>
#include <string>

using Configurations = std::pair<std::string, std::vector<size_t>>;
using Parameters = std::map<std::string, size_t>;

template <typename net_t> class OpenCL;

template <typename net_t>
class Tuner {
    OpenCL<net_t> & m_opencl;
    cl::Context m_context;
    cl::Device m_device;
public:
    std::string tune_sgemm(const int m, const int n, const int k,
                           const int batch_size, const int runs = 4);
    std::string load_sgemm_tuners(const int m, const int n, const int k,
                                  const int batch_size);

    // list of device types that was tuned in this run.
    // This is to prevent the same device from being tuned multiple times.
    static std::vector<std::string> tuned_devices;

    static constexpr auto TUNER_VERSION = 0;
    Tuner(OpenCL<net_t> & opencl, cl::Context context, cl::Device device) :
        m_opencl(opencl), m_context(context), m_device(device) {}
private:
    void store_sgemm_tuners(const int m, const int n, const int k,
                            const int batch_size, std::string tuners);
    bool valid_config_sgemm(Parameters p, bool exhaustive);
    std::string parameters_to_defines(const Parameters& p);
    std::string parameters_to_string(const Parameters& p);
    Parameters get_parameters_by_int(const std::vector<Configurations>& opts,
                                     const int n);
    std::string sgemm_tuners_from_line(std::string line, const int m,
                                       const int n, const int k,
                                       const int batch_size);
};

#endif
