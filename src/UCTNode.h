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

#include "GameState.h"
#include "Network.h"
#include "SMP.h"

class UCTNode;

class UCTEdge {
public:
    using node_ptr_t = std::unique_ptr<UCTNode>;

    explicit UCTEdge(int vertex, float score);
    UCTEdge() = delete;
    ~UCTEdge() = default;
    int get_move() const;
    float get_score() const;
    void set_score(float score);
    UCTNode* get_child() const;
    void invalidate();
    bool valid() const;
    int get_visits() const;
    float get_eval(int tomove) const;
    UCTNode* get_or_create_child(std::atomic<int>& nodecount);
    node_ptr_t pass_or_create_child(std::atomic<int>& nodecount);
    bool first_visit() const;

private:
    void create_child(std::atomic<int>& nodecount);
    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.
    std::int16_t m_move;
    std::atomic<bool> m_valid{true};  // Validity under superko rules.
    SMP::Mutex m_nodemutex;  // Guards the creation of nodes.
    float m_score;
    node_ptr_t m_child;
};

class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;

    using edge_ptr_t = std::unique_ptr<UCTEdge>;

    explicit UCTNode(int vertex, float score);
    UCTNode() = delete;
    ~UCTNode() = default;
    bool first_visit() const;
    bool has_edges() const;
    bool create_edges(std::atomic<int>& edgecount,
                      GameState& state, float& eval);
    float eval_state(GameState& state);
    void kill_superkos(const KoState& state);
    void invalidate();
    bool valid() const;
    int get_move() const;
    int get_visits() const;
    float get_score() const;
    void set_score(float score);
    float get_eval(int tomove) const;
    float get_net_eval(int tomove) const;
    double get_blackevals() const;
    void accumulate_eval(float eval);
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void dirichlet_noise(float epsilon, float alpha);
    void randomize_first_proportionally();
    void update(float eval);

    UCTEdge* uct_select_edge(int color);
    UCTEdge* get_first_edge() const;
    UCTEdge* get_nopass_edge(FastState& state) const;
    const std::vector<edge_ptr_t>& get_edges() const;
    size_t count_nodes() const;
    size_t count_edges() const;
    UCTEdge* find_edge(const int move);
    void sort_edges(int color);
    UCTEdge* get_best_root_edge(int color);
    SMP::Mutex& get_mutex();

private:
    void link_nodelist(std::atomic<int>& edgecount,
                       std::vector<Network::scored_node>& nodelist);
    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_score;
    float m_net_eval{0};  // Original net eval for this node (not children).
    std::atomic<double> m_blackevals{0};
    // node alive (not superko)
    std::atomic<bool> m_valid{true};
    // Is someone adding scores to this node?
    // We don't need to unset this.
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<bool> m_has_edges{false};
    std::vector<edge_ptr_t> m_edges;
};

#endif
