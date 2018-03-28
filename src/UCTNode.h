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

#ifndef UCTNODE_H_INCLUDED
#define UCTNODE_H_INCLUDED

#include "config.h"

#include <atomic>
#include <memory>
#include <vector>
#include <cassert>
#include <cstring>

#include "GameState.h"
#include "Network.h"
#include "SMP.h"

class UCTNode;

// 'lazy-initializable' version of std::unique_ptr<UCTNode>.
// When a UCTNodePointer is constructed, the constructor arguments are stored instead
// of constructing the actual UCTNode instance.  Later when the UCTNode is needed,
// the external code calls inflate() which actually constructs the UCTNode.
// Basically, this is a 'tagged union' of:
//  - std::unique_ptr<UCTNode> pointer;
//  - std::pair<float, std::int16_t> args;

// WARNING : inflate() is not thread-safe and hence has to be protected by an external lock.

class UCTNodePointer {
private:
    // the raw storage used here.
    // if bit 0 is 0, m_data is the actual pointer.
    // if bit 0 is 1, bit [31:16] is the vertex value, bit [63:32] is the score.
    // (I really wanted to use C-style bit fields and unions, but those aren't portable)
    mutable uint64_t m_data = 1;


    UCTNode * read_ptr() const {
        assert(is_inflated());
        return reinterpret_cast<UCTNode*>(m_data);
    }

    std::int16_t read_vertex() const {
        assert(!is_inflated());
        return static_cast<std::int16_t>(m_data >> 16);
    }
    float read_score() const {
        static_assert(sizeof(float) == 4, "This code assumes floats are 32-bit");
        assert(!is_inflated());

        auto x = static_cast<std::uint32_t>(m_data >> 32);
        float ret;
        std::memcpy(&ret, &x, sizeof(ret));
        return ret;
    }
public:
    ~UCTNodePointer();
    UCTNodePointer(UCTNodePointer&& n);
    UCTNodePointer(std::int16_t vertex, float score);
    UCTNodePointer(const UCTNodePointer&) = delete;

    bool is_inflated() const {
        return (m_data & 1ULL) == 0;
    } 

    // methods from std::unique_ptr<UCTNode>
    typename std::add_lvalue_reference<UCTNode>::type operator*() const{
        return *read_ptr();
    }
    UCTNode* operator->() const {
        return read_ptr();
    }
    UCTNode* get() const {
        return read_ptr();
    }
    UCTNodePointer& operator=(UCTNodePointer&& n);
    UCTNode * release() {
        auto ret = read_ptr();
        m_data = 1;
        return ret;
    }

    // construct UCTNode instance from the vertex/score pair
    void inflate() const;

    // proxy of UCTNode methods which can be called without constructing UCTNode
    bool valid() const;
    int get_visits() const;
    float get_score() const;
    bool active() const;
    int get_move() const;
    // this can only be called if it is an inflated pointer
    float get_eval(int tomove) const; 
};


class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;
    // Defined in UCTNode.cpp
    explicit UCTNode(int vertex, float score);
    UCTNode() = delete;
    ~UCTNode() = default;

    bool create_children(std::atomic<int>& nodecount,
                         GameState& state, float& eval,
                         float mem_full = 0.0f);

    const std::vector<UCTNodePointer>& get_children() const;
    void sort_children(int color);
    UCTNode& get_best_root_child(int color);
    UCTNode* uct_select_child(int color, bool is_root);

    size_t count_nodes() const;
    SMP::Mutex& get_mutex();
    bool first_visit() const;
    bool has_children() const;
    void invalidate();
    void set_active(const bool active);
    bool valid() const;
    bool active() const;
    int get_move() const;
    int get_visits() const;
    float get_score() const;
    void set_score(float score);
    float get_eval(int tomove) const;
    float get_net_eval(int tomove) const;
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void update(float eval);

    // Defined in UCTNodeRoot.cpp, only to be called on m_root in UCTSearch
    void kill_superkos(const KoState& state);
    void dirichlet_noise(float epsilon, float alpha);
    void randomize_first_proportionally();

    UCTNode* get_first_child() const;
    UCTNode* get_nopass_child(FastState& state) const;
    std::unique_ptr<UCTNode> find_child(const int move);
    
    void inflate_all_children();


private:
    enum Status : char {
        INVALID, // superko
        PRUNED,
        ACTIVE
    };
    void link_nodelist(std::atomic<int>& nodecount,
                       std::vector<Network::scored_node>& nodelist,
                       float mem_full);
    double get_blackevals() const;
    void accumulate_eval(float eval);

    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.

    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_score;
    // Original net eval for this node (not children).
    float m_net_eval{0.0f};
    std::atomic<double> m_blackevals{0.0};
    std::atomic<Status> m_status{ACTIVE};
    // Is someone adding scores to this node?
    // We don't need to unset this.
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<bool> m_has_children{false};
    std::vector<UCTNodePointer> m_children;
};


inline UCTNodePointer::~UCTNodePointer() {
    if (is_inflated()) {
        delete read_ptr();
    }
}

inline UCTNodePointer::UCTNodePointer(UCTNodePointer&& n) {
    if (is_inflated()) {
        delete read_ptr();
    }
    m_data = n.m_data;
    n.m_data = 1; // non-inflated garbage
}

inline UCTNodePointer::UCTNodePointer(std::int16_t vertex, float score) {
    std::uint32_t i_score;
    auto i_vertex = static_cast<std::uint16_t>(vertex);
    std::memcpy(&i_score, &score, sizeof(i_score));
    
    m_data = (static_cast<std::uint64_t>(i_score) << 32) | (static_cast<std::uint64_t>(i_vertex) << 16) | 1ULL;
}

inline UCTNodePointer& UCTNodePointer::operator=(UCTNodePointer&& n) {
    if ( is_inflated() ) {
        delete read_ptr();
    }
    m_data = n.m_data;
    n.m_data = 1;
    
    return *this;
}

inline void UCTNodePointer::inflate() const {
    if (is_inflated()) return;

    m_data = reinterpret_cast<std::uint64_t>(new UCTNode(read_vertex(), read_score()));
}
    
inline bool UCTNodePointer::valid() const {
    if (is_inflated()) return read_ptr()->valid();
    return true;
}
inline int UCTNodePointer::get_visits() const {
    if (is_inflated()) return read_ptr()->get_visits();
    return 0;
}
inline float UCTNodePointer::get_score() const {
    if (is_inflated()) return read_ptr()->get_score();
    return read_score();
}
inline bool UCTNodePointer::active() const {
    if (is_inflated()) return read_ptr()->active();
    return true;
}
inline float UCTNodePointer::get_eval(int tomove) const {
    // this can only be called if it is an inflated pointer
    assert(is_inflated());
    return read_ptr()->get_eval(tomove);
}
inline int UCTNodePointer::get_move() const {
    if (is_inflated()) return read_ptr()->get_move();
    return read_vertex();
}
#endif
