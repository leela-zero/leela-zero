/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Gian-Carlo Pascutto

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

#ifndef UCTNODEPOINTER_H_INCLUDED
#define UCTNODEPOINTER_H_INCLUDED

#include "config.h"

#include <atomic>
#include <memory>
#include <cassert>
#include <cstring>

#include "SMP.h"

class UCTNode;

// 'lazy-initializable' version of std::unique_ptr<UCTNode>.
// When a UCTNodePointer is constructed, the constructor arguments
// are stored instead of constructing the actual UCTNode instance.
// Later when the UCTNode is needed, the external code calls inflate()
// which actually constructs the UCTNode. Basically, this is a 'tagged union'
// of:
//  - std::unique_ptr<UCTNode> pointer;
//  - std::pair<float, std::int16_t> args;

// All methods should be thread-safe except destructor and when
// the instanced is 'moved from'.

class UCTNodePointer {
private:
    static constexpr std::uint64_t INVALID = 2;
    static constexpr std::uint64_t POINTER = 1;
    static constexpr std::uint64_t UNINFLATED = 0;

    static std::atomic<size_t> m_tree_size;
    static void increment_tree_size(size_t sz);
    static void decrement_tree_size(size_t sz);

    // the raw storage used here.
    // if bit [1:0] is 1, m_data is the actual pointer.
    // if bit [1:0] is 0, bit [31:16] is the vertex value, bit [63:32] is the policy
    // if bit [1:0] is other values, it should assert-fail
    // (C-style bit fields and unions are not portable)
    mutable std::atomic<std::uint64_t> m_data{INVALID};

    UCTNode * read_ptr(uint64_t v) const {
        assert((v & 3ULL) == POINTER);
        return reinterpret_cast<UCTNode*>(v & ~(0x3ULL));
    }

    std::int16_t read_vertex(uint64_t v) const {
        assert((v & 3ULL) == UNINFLATED);
        return static_cast<std::int16_t>(v >> 16);
    }

    float read_policy(uint64_t v) const {
        static_assert(sizeof(float) == 4,
            "This code assumes floats are 32-bit");
        assert((v & 3ULL) == UNINFLATED);

        auto x = static_cast<std::uint32_t>(v >> 32);
        float ret;
        std::memcpy(&ret, &x, sizeof(ret));
        return ret;
    }

    bool is_inflated(uint64_t v) const {
        return (v & 3ULL) == POINTER;
    }

public:
    static size_t get_tree_size();

    ~UCTNodePointer();
    UCTNodePointer(UCTNodePointer&& n);
    UCTNodePointer(std::int16_t vertex, float policy);
    UCTNodePointer(const UCTNodePointer&) = delete;


    bool is_inflated() const {
        return is_inflated(m_data.load());
    }

    // methods from std::unique_ptr<UCTNode>
    typename std::add_lvalue_reference<UCTNode>::type operator*() const{
        return *read_ptr(m_data.load());
    }
    UCTNode* operator->() const {
        return read_ptr(m_data.load());
    }
    UCTNode* get() const {
        return read_ptr(m_data.load());
    }
    UCTNodePointer& operator=(UCTNodePointer&& n);
    UCTNode * release();

    // construct UCTNode instance from the vertex/policy pair
    void inflate() const;

    // proxy of UCTNode methods which can be called without
    // constructing UCTNode
    bool valid() const;
    int get_visits() const;
    float get_policy() const;
    bool active() const;
    int get_move() const;
    // these can only be called if it is an inflated pointer
    float get_eval(int tomove) const;
    float get_eval_lcb(int color) const;
};

#endif
