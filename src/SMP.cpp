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

#include "SMP.h"

#include <cassert>
#include <thread>


SMP::Mutex::Mutex() {
    m_lock = false;
}

SMP::Lock::Lock(Mutex & m) {
    m_mutex = &m;
    lock();
}

void SMP::Lock::lock() {
    assert(!m_owns_lock);
    // Test and Test-and-Set reduces memory contention
    // However, just trying to Test-and-Set first improves performance in almost
    // all cases
    while (m_mutex->m_lock.exchange(true, std::memory_order_acquire)) {
      while (m_mutex->m_lock.load(std::memory_order_relaxed));
    }
    m_owns_lock = true;
}

void SMP::Lock::unlock() {
    assert(m_owns_lock);
    auto lock_held = m_mutex->m_lock.exchange(false, std::memory_order_release);

    // If this fails it means we are unlocking an unlocked lock
#ifdef NDEBUG
    (void)lock_held;
#else
    assert(lock_held);
#endif
    m_owns_lock = false;
}

SMP::Lock::~Lock() {
    // If we don't claim to hold the lock,
    // don't bother trying to unlock in the destructor.
    if (m_owns_lock) {
        unlock();
    }
}


SMP::RWMutex::RWMutex() {
    m_lock = 0;
    m_write_sequence = 0;
}

SMP::RWLock::RWLock(RWMutex & m, bool is_writelock) {
    m_mutex = &m;
    if (is_writelock) {
        wlock();
    } else {
        rlock();
    }
}

void SMP::RWLock::wlock() {
    assert(!m_owns_wlock);
    assert(!m_owns_rlock);
    int16_t expected = 0;
    while (!m_mutex->m_lock.compare_exchange_strong(expected, -1)) {
        expected = 0;
    }
    m_mutex->m_write_sequence++;
    m_owns_rlock = false;
    m_owns_wlock = true;
}


void SMP::RWLock::rlock() {
    assert(!m_owns_rlock);
    assert(!m_owns_wlock);
    while (true) {
        auto expected = m_mutex->m_lock.load();
        auto newval = expected + 1;
        if (expected >= 0 &&
            m_mutex->m_lock.compare_exchange_strong(expected, newval) )
        {
            break;
        }
    }
    m_owns_rlock = true;
    m_owns_wlock = false;
}

void SMP::RWLock::unlock() {
    assert(m_owns_rlock || m_owns_wlock);
    if (m_owns_rlock) {
        auto v = --m_mutex->m_lock;
        assert(v >= 0);
        m_owns_rlock = 0;

#ifdef NDEBUG
        (void) v;
#endif
    }
    if (m_owns_wlock) {
        auto v = ++m_mutex->m_lock;
        assert(v == 0);
        m_owns_wlock = 0;

#ifdef NDEBUG
        (void) v;
#endif
    }

}

bool SMP::RWLock::try_wlock() {
    assert(m_owns_rlock);

    auto v = m_mutex->m_write_sequence;
    unlock();
    wlock();
    if (v + 1 != m_mutex->m_write_sequence) {
        unlock();
        return false;
    } else {
        return true;
    }
}

SMP::RWLock::~RWLock() {
    // If we don't claim to hold the lock,
    // don't bother trying to unlock in the destructor.
    if (m_owns_rlock || m_owns_wlock) {
        unlock();
    }
}

int SMP::get_num_cpus() {
    return std::thread::hardware_concurrency();
}
