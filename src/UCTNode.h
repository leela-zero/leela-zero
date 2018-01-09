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
#include <limits>
#include <memory>
#include <vector>

#include "GameState.h"
#include "Network.h"
#include "SMP.h"

class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;

    using node_ptr_t = std::unique_ptr<UCTNode>;

    explicit UCTNode(int vertex, float score, float init_eval);
    UCTNode() = delete;
    ~UCTNode() = default;
    bool first_visit() const;
    bool has_children() const;
    bool create_children(std::atomic<int>& nodecount,
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
    double get_blackevals() const;
    void set_visits(int visits);
    void set_blackevals(double blacevals);
    void accumulate_eval(float eval);
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void dirichlet_noise(float epsilon, float alpha);
    void randomize_first_proportionally();
    void update(float eval = std::numeric_limits<float>::quiet_NaN());

    UCTNode* uct_select_child(int color);
    UCTNode* get_first_child() const;
    UCTNode* get_nopass_child(FastState& state) const;
    const std::vector<node_ptr_t>& get_children() const;

    void sort_root_children(int color);
    UCTNode& get_best_root_child(int color);
    SMP::Mutex& get_mutex();

private:
    void link_nodelist(std::atomic<int>& nodecount,
                       std::vector<Network::scored_node>& nodelist,
                       float init_eval);
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
    float m_init_eval;
    std::atomic<double> m_blackevals{0};
    // node alive (not superko)
    std::atomic<bool> m_valid{true};
    // Is someone adding scores to this node?
    // We don't need to unset this.
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<bool> m_has_children{false};
    std::vector<node_ptr_t> m_children;
};

#endif
