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

#include "config.h"

#include <assert.h>
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "KoState.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float score, float init_eval)
    : m_move(vertex), m_score(score), m_init_eval(init_eval) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int> & nodecount,
                              GameState & state,
                              float & eval) {
    // check whether somebody beat us to it (atomic)
    if (has_children()) {
        return false;
    }
    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }
    // check whether somebody beat us to it (after taking the lock)
    if (has_children()) {
        return false;
    }
    // Someone else is running the expansion
    if (m_is_expanding) {
        return false;
    }
    // We'll be the one queueing this node for expansion, stop others
    m_is_expanding = true;
    lock.unlock();

    const auto raw_netlist = Network::get_scored_moves(
        &state, Network::Ensemble::RANDOM_ROTATION);

    // DCNN returns winrate as side to move
    auto net_eval = raw_netlist.second;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        net_eval = 1.0f - net_eval;
    }
    eval = net_eval;

    std::vector<Network::scored_node> nodelist;

    auto legal_sum = 0.0f;
    for (const auto& node : raw_netlist.first) {
        auto vertex = node.second;
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(node);
            legal_sum += node.first;
        }
    }

    // If the sum is 0 or a denormal, then don't try to normalize.
    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    }

    link_nodelist(nodecount, nodelist, net_eval);
    return true;
}

void UCTNode::link_nodelist(std::atomic<int> & nodecount,
                            std::vector<Network::scored_node> & nodelist,
                            float init_eval) {
    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(get_mutex(), lock);

    m_children.reserve(nodelist.size());
    for (const auto& node : nodelist) {
        m_children.emplace_back(
            std::make_unique<UCTNode>(node.second, node.first, init_eval)
        );
    }

    nodecount += m_children.size();
    m_has_children = true;
}

void UCTNode::kill_superkos(const KoState& state) {
    for (auto& child : m_children) {
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            KoState mystate = state;
            mystate.play_move(move);

            if (mystate.superko()) {
                // Don't delete nodes for now, just mark them invalid.
                child->invalidate();
            }
        }
    }

    // Now do the actual deletion.
    m_children.erase(
        std::remove_if(begin(m_children), end(m_children),
                       [](const auto &child) { return !child->valid(); }),
        end(m_children)
    );
}

float UCTNode::eval_state(GameState& state) {
    auto raw_netlist = Network::get_scored_moves(
        &state, Network::Ensemble::RANDOM_ROTATION, -1, true);

    // DCNN returns winrate as side to move
    auto net_eval = raw_netlist.second;

    // But we score from black's point of view
    if (state.board.white_to_move()) {
        net_eval = 1.0f - net_eval;
    }

    return net_eval;
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    auto child_cnt = m_children.size();

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::get_Rng()));
    }

    auto sample_sum = std::accumulate(begin(dirichlet_vector),
                                      end(dirichlet_vector), 0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        return;
    }

    for (auto& v: dirichlet_vector) {
        v /= sample_sum;
    }

    child_cnt = 0;
    for (auto& child : m_children) {
        auto score = child->get_score();
        auto eta_a = dirichlet_vector[child_cnt++];
        score = score * (1 - epsilon) + epsilon * eta_a;
        child->set_score(score);
    }
}

void UCTNode::randomize_first_proportionally() {
    auto accum = std::uint32_t{0};
    auto accum_vector = std::vector<decltype(accum)>{};
    for (const auto& child : m_children) {
        accum += child->get_visits();
        accum_vector.emplace_back(accum);
    }

    auto pick = Random::get_Rng().randuint32(accum);
    auto index = size_t{0};
    for (size_t i = 0; i < accum_vector.size(); i++) {
        if (pick < accum_vector[i]) {
            index = i;
            break;
        }
    }

    // Take the early out
    if (index == 0) {
        return;
    }

    assert(m_children.size() >= index);

    // Now swap the child at index with the first child
    std::iter_swap(begin(m_children), begin(m_children) + index);
}

int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float eval) {
    m_visits++;
    accumulate_eval(eval);
}

bool UCTNode::has_children() const {
    return m_has_children;
}

void UCTNode::set_visits(int visits) {
    m_visits = visits;
}

float UCTNode::get_score() const {
    return m_score;
}

void UCTNode::set_score(float score) {
    m_score = score;
}

int UCTNode::get_visits() const {
    return m_visits;
}

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    if (visits > 0) {
        auto blackeval = get_blackevals();
        if (tomove == FastBoard::WHITE) {
            blackeval += static_cast<double>(virtual_loss);
        }
        auto score = static_cast<float>(blackeval / (double)visits);
        if (tomove == FastBoard::WHITE) {
            score = 1.0f - score;
        }
        return score;
    } else {
        // If a node has not been visited yet,
        // the eval is that of the parent.
        auto eval = m_init_eval;
        if (tomove == FastBoard::WHITE) {
            eval = 1.0f - eval;
        }
        return eval;
    }
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::set_blackevals(double blackevals) {
    m_blackevals = blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, (double)eval);
}

UCTNode* UCTNode::uct_select_child(int color) {
    UCTNode* best = nullptr;
    auto best_value = -1000.0f;

    LOCK(get_mutex(), lock);

    // Count parentvisits.
    // We do this manually to avoid issues with transpositions.
    auto parentvisits = size_t{0};
    for (const auto& child : m_children) {
        if (child->valid()) {
            parentvisits += child->get_visits();
        }
    }
    auto numerator = static_cast<float>(std::sqrt((double)parentvisits));

    for (const auto& child : m_children) {
        if (!child->valid()) {
            continue;
        }

        // get_eval() will automatically set first-play-urgency
        auto winrate = child->get_eval(color);
        auto psa = child->get_score();
        auto denom = 1.0f + child->get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        auto value = winrate + puct;
        assert(value > -1000.0f);

        if (value > best_value) {
            best_value = value;
            best = child.get();
        }
    }

    assert(best != nullptr);
    return best;
}

class NodeComp : public std::binary_function<UCTNode::node_ptr_t&,
                                             UCTNode::node_ptr_t&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNode::node_ptr_t& a,
                    const UCTNode::node_ptr_t& b) {
        // if visits are not same, sort on visits
        if (a->get_visits() != b->get_visits()) {
            return a->get_visits() < b->get_visits();
        }

        // neither has visits, sort on prior score
        if (a->get_visits() == 0) {
            return a->get_score() < b->get_score();
        }

        // both have same non-zero number of visits
        return a->get_eval(m_color) < b->get_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_root_children(int color) {
    LOCK(get_mutex(), lock);
    std::stable_sort(begin(m_children), end(m_children), NodeComp(color));
    std::reverse(begin(m_children), end(m_children));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_children.empty());

    return *(std::max_element(begin(m_children), end(m_children),
                              NodeComp(color))->get());
}

UCTNode* UCTNode::get_first_child() const {
    if (m_children.empty()) {
        return nullptr;
    }
    return m_children.front().get();
}

const std::vector<UCTNode::node_ptr_t>& UCTNode::get_children() const {
    return m_children;
}

UCTNode* UCTNode::get_nopass_child(FastState& state) const {
    for (const auto& child : m_children) {
        /* If we prevent the engine from passing, we must bail out when
           we only have unreasonable moves to pick, like filling eyes.
           Note that this isn't knowledge isn't required by the engine,
           we require it because we're overruling its moves. */
        if (child->m_move != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), child->m_move)) {
            return child.get();
        }
    }
    return nullptr;
}

void UCTNode::invalidate() {
    m_valid = false;
}

bool UCTNode::valid() const {
    return m_valid;
}
