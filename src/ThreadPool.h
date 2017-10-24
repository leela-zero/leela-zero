#ifndef THREADPOOL_H_INCLUDED
#define THREADPOOL_H_INCLUDED
/*
    Extended from code:
    Copyright (c) 2012 Jakob Progsch, Václav Zeman

    This software is provided 'as-is', without any express or implied
    warranty. In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

#include <cstddef>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <future>
#include <functional>

namespace Utils {

class ThreadPool {
public:
    ThreadPool() = default;
    ~ThreadPool();
    void initialize(std::size_t);
    template<class F, class... Args>
    auto add_task(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
private:
    std::vector<std::thread> m_threads;
    std::queue<std::function<void()>> m_tasks;

    std::mutex m_mutex;
    std::condition_variable m_condvar;
    bool m_exit{false};
};

inline void ThreadPool::initialize(size_t threads) {
    for (size_t i = 0; i < threads; i++) {
        m_threads.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_condvar.wait(lock, [this]{ return m_exit || !m_tasks.empty(); });
                    if (m_exit && m_tasks.empty()) {
                        return;
                    }
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }
                task();
            }
        });
    }
}

template<class F, class... Args>
auto ThreadPool::add_task(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_tasks.emplace([task](){(*task)();});
    }
    m_condvar.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_exit = true;
    }
    m_condvar.notify_all();
    for (std::thread & worker: m_threads) {
        worker.join();
    }
}

class ThreadGroup {
public:
    ThreadGroup(ThreadPool & pool) : m_pool(pool) {};
    template<class F, class... Args>
    void add_task(F&& f, Args&&... args) {
        m_taskresults.emplace_back(
            m_pool.add_task(std::forward<F>(f), std::forward<Args>(args)...)
        );
    };
    void wait_all() {
        for (auto && result: m_taskresults) {
            result.get();
        }
    };
private:
    ThreadPool & m_pool;
    std::vector<std::future<void>> m_taskresults;
};

}

#endif