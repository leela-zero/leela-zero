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

#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"
#include <string>
#include <atomic>
#include "ThreadPool.h"

#ifdef _MSC_VER
#define ASSUME_ALIGNED(p, n) \
__assume((reinterpret_cast<std::size_t>(p) & ((n) - 1)) == 0)
#else
#define ASSUME_ALIGNED(p, n) \
(p) = static_cast<__typeof__(p)>(__builtin_assume_aligned((p), (n)))
#endif

extern Utils::ThreadPool thread_pool;

namespace Utils {
    void myprintf(const char *fmt, ...);
    void gtp_printf(int id, const char *fmt, ...);
    void gtp_fail_printf(int id, const char *fmt, ...);
    void log_input(std::string input);
    bool input_pending();
    bool input_causes_stop();

    template<class T>
    void atomic_add(std::atomic<T> &f, T d) {
        T old = f.load();
        while (!f.compare_exchange_weak(old, old + d));
    }

    template<class T>
    bool is_aligned(T* ptr, size_t alignment) {
        return (uintptr_t(ptr) & (alignment - 1)) == 0;
    }

    template<typename T>
    T rotl(const T x, const int k) {
	    return (x << k) | (x >> (std::numeric_limits<T>::digits - k));
    }
}

#endif
