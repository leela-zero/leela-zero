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
#include "UCTNodePointer.h"

class UCTSearch;

class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;
    // Defined in UCTNode.cpp
    explicit UCTNode(int vertex, float policy);
    UCTNode() = delete;
    ~UCTNode() = default;

    bool create_children(Network & network,
                         std::atomic<int>& nodecount,
                         GameState& state, float& eval,
                         float min_psa_ratio = 0.0f,
                         int symmetry = -1);

    const std::vector<UCTNodePointer>& get_children() const;
    void sort_children(int color);
    UCTNode& get_best_root_child(int color);
    UCTNode* uct_select_child(int color, bool is_root);

    size_t count_nodes() const;
    SMP::Mutex& get_mutex();
    bool first_visit() const;
    bool has_children() const;
    bool expandable(const float min_psa_ratio = 0.0f) const;
    void invalidate();
    void set_active(const bool active);
    bool valid() const;
    bool active() const;
    int get_move() const;
    int get_visits() const;
    float get_policy() const;
    void set_policy(float policy);
    float get_eval(int tomove) const;
    float get_raw_eval(int tomove, int virtual_loss = 0) const;
    float get_net_eval(int tomove) const;
    void set_net_eval(float eval) { m_net_eval = eval; }
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void update(float eval);
    void clear(Network & net, std::atomic<int>& nodes, GameState& root_state, float& eval);

    // Defined in UCTNodeRoot.cpp, only to be called on m_root in UCTSearch
    void randomize_first_proportionally();
    void prepare_root_node(Network & network, int color,
                           std::atomic<int>& nodecount,
                           GameState& state, UCTSearch * search);

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
                       std::vector<Network::PolicyVertexPair>& nodelist,
                       float min_psa_ratio);
    double get_blackevals() const;
    void accumulate_eval(float eval);
    void kill_superkos(const KoState& state);
    void dirichlet_noise(float epsilon, float alpha);

    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.

    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_policy;
    // Original net eval for this node (not children).
    float m_net_eval{2.0f};
    std::atomic<double> m_blackevals{0.0};
    std::atomic<Status> m_status{ACTIVE};
    // Is someone adding policy priors to this node?
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<float> m_min_psa_ratio_children{2.0f};
    std::vector<UCTNodePointer> m_children;
};

#endif
