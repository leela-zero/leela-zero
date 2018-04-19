/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Gian-Carlo Pascutto

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

#include <atomic>
#include <memory>
#include <cassert>
#include <cstring>

#include "UCTNode.h"

UCTNodePointer::~UCTNodePointer() {
    auto v = std::atomic_exchange(&m_data, INVALID);
    if (is_inflated(v)) {
        delete read_ptr(v);
    }
}

UCTNodePointer::UCTNodePointer(UCTNodePointer&& n) {
    auto nv = std::atomic_exchange(&n.m_data, INVALID);
    auto v = std::atomic_exchange(&m_data, nv);
    if (is_inflated(v)) {
        delete read_ptr(v);
    }
}

UCTNodePointer::UCTNodePointer(std::int16_t vertex, float score) {
    std::uint32_t i_score;
    auto i_vertex = static_cast<std::uint16_t>(vertex);
    std::memcpy(&i_score, &score, sizeof(i_score));

    m_data =  (static_cast<std::uint64_t>(i_score)  << 32)
            | (static_cast<std::uint64_t>(i_vertex) << 16);
}

UCTNodePointer& UCTNodePointer::operator=(UCTNodePointer&& n) {
    auto nv = std::atomic_exchange(&n.m_data, INVALID);
    auto v = std::atomic_exchange(&m_data, nv);

    if (is_inflated(v)) {
        delete read_ptr(v);
    }
    return *this;
}

void UCTNodePointer::inflate() const {
    while(true) {
        auto v = m_data.load();
        if (is_inflated(v)) return;

        auto v2 = reinterpret_cast<std::uint64_t>(
            new UCTNode(read_vertex(v), read_score(v))
        ) | POINTER;
        bool success = m_data.compare_exchange_strong(v, v2);
        if (success) {
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

float UCTNodePointer::get_score() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_score();
    return read_score(v);
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
