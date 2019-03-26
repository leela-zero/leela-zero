/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Gian-Carlo Pascutto and contributors

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

#include <atomic>
#include <memory>
#include <cassert>
#include <cstring>

#include "UCTNode.h"

std::atomic<size_t> UCTNodePointer::m_tree_size = {0};

size_t UCTNodePointer::get_tree_size() {
    return m_tree_size.load();
}

void UCTNodePointer::increment_tree_size(size_t sz) {
    m_tree_size += sz;
}

void UCTNodePointer::decrement_tree_size(size_t sz) {
    assert(UCTNodePointer::m_tree_size >= sz);
    m_tree_size -= sz;
}

UCTNodePointer::~UCTNodePointer() {
    auto sz = sizeof(UCTNodePointer);
    auto v = m_data.load();
    if (is_inflated(v)) {
        delete read_ptr(v);
        sz += sizeof(UCTNode);
    }
    decrement_tree_size(sz);
}

UCTNodePointer::UCTNodePointer(UCTNodePointer&& n) {
    auto nv = std::atomic_exchange(&n.m_data, INVALID);
    auto v = std::atomic_exchange(&m_data, nv);
#ifdef NDEBUG
    (void)v;
#else
    assert(v == INVALID);
#endif
    increment_tree_size(sizeof(UCTNodePointer));
}

UCTNodePointer::UCTNodePointer(std::int16_t vertex, float policy) {
    std::uint32_t i_policy;
    auto i_vertex = static_cast<std::uint16_t>(vertex);
    std::memcpy(&i_policy, &policy, sizeof(i_policy));

    m_data =  (static_cast<std::uint64_t>(i_policy)  << 32)
            | (static_cast<std::uint64_t>(i_vertex) << 16);
    increment_tree_size(sizeof(UCTNodePointer));
}

UCTNodePointer& UCTNodePointer::operator=(UCTNodePointer&& n) {
    auto nv = std::atomic_exchange(&n.m_data, INVALID);
    auto v = std::atomic_exchange(&m_data, nv);

    if (is_inflated(v)) {
        decrement_tree_size(sizeof(UCTNode));
        delete read_ptr(v);
    }
    return *this;
}

UCTNode * UCTNodePointer::release() {
    auto v = std::atomic_exchange(&m_data, INVALID);
    decrement_tree_size(sizeof(UCTNode));
    return read_ptr(v);
}

void UCTNodePointer::inflate() const {
    while (true) {
        auto v = m_data.load();
        if (is_inflated(v)) return;

        auto v2 = reinterpret_cast<std::uint64_t>(
            new UCTNode(read_vertex(v), read_policy(v))
        ) | POINTER;
        bool success = m_data.compare_exchange_strong(v, v2);
        if (success) {
            increment_tree_size(sizeof(UCTNode));
            return;
        } else {
            // this means that somebody else also modified this instance.
            // Try again next time
            delete read_ptr(v2);
        }
    }
}

bool UCTNodePointer::valid() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->valid();
    return true;
}

int UCTNodePointer::get_visits() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_visits();
    return 0;
}

float UCTNodePointer::get_policy() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_policy();
    return read_policy(v);
}

float UCTNodePointer::get_eval_lcb(int color) const {
    assert(is_inflated());
    auto v = m_data.load();
    return read_ptr(v)->get_eval_lcb(color);
}

bool UCTNodePointer::active() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->active();
    return true;
}

float UCTNodePointer::get_eval(int tomove) const {
    // this can only be called if it is an inflated pointer
    auto v = m_data.load();
    assert(is_inflated(v));
    return read_ptr(v)->get_eval(tomove);
}

int UCTNodePointer::get_move() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_move();
    return read_vertex(v);
}
