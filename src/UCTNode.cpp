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

#include <cassert>
#include <cstdio>
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

UCTNode::UCTNode(int vertex, float score) : m_move(vertex), m_score(score) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int>& score_count,
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
    m_net_eval = raw_netlist.second;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        m_net_eval = 1.0f - m_net_eval;
    }
    eval = m_net_eval;

    std::vector<Network::scored_node> nodelist;

    auto legal_sum = 0.0f;
    for (const auto& node : raw_netlist.first) {
        auto vertex = node.second;
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(node);
            legal_sum += node.first;
        }
    }

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    assert(!nodelist.empty());

    m_child_scores.reserve(nodelist.size());
    for (const auto& node : nodelist) {
        m_child_scores.emplace_back(node.second, node.first);
    }

    m_has_children = true;
    score_count += nodelist.size();

    return true;
}

// Only safe to call in single threaded context after calling expand_all
void UCTNode::kill_superkos(const KoState& state) {
    LOCK(get_mutex(), lock);
    assert(m_child_scores.empty());

    std::vector<node_ptr_t> good_expanded;
    good_expanded.reserve(m_expanded.size());
    for (auto& child : m_expanded) {
        const auto move = child->get_move();
        if (move != FastBoard::PASS) {
            KoState mystate = state;
            mystate.play_move(move);

            if (mystate.superko()) {
                // Skip invalid moves.
                continue;
            }
        }
        good_expanded.emplace_back(std::move(child));
    }

    // Now do the actual deletions
    std::swap(m_expanded, good_expanded);
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

// Should only be called once after expand_all
void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    LOCK(get_mutex(), lock);
    assert(m_child_scores.empty());

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < m_expanded.size(); i++) {
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

    auto child_cnt = 0;
    for (auto& child : m_expanded) {
        auto score = child->get_score();
        auto eta_a = dirichlet_vector[child_cnt++];

        score = score * (1 - epsilon) + epsilon * eta_a;
        child->set_score(score);
    }
}

void UCTNode::randomize_first_proportionally() {
    LOCK(get_mutex(), lock);

    auto accum = std::uint64_t{0};
    auto accum_vector = std::vector<decltype(accum)>{};
    // Only need to consider m_expanded as nothing else has any visits.
    for (const auto& child : m_expanded) {
        accum += child->get_visits();
        accum_vector.emplace_back(accum);
    }

    auto pick = Random::get_Rng().randuint64(accum);
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

    assert(m_expanded.size() > index);

    // Now swap the child at index with the first child
    std::iter_swap(begin(m_expanded), begin(m_expanded) + index);
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
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto score = static_cast<float>(blackeval / (double)visits);
    if (tomove == FastBoard::WHITE) {
        score = 1.0f - score;
    }
    return score;
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_eval;
    }
    return m_net_eval;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, (double)eval);
}

// Expand nth child and remove from m_child_scores.
UCTNode* UCTNode::expand(size_t child) {
    // Relies on caller to hold a lockode.
    assert(child < m_child_scores.size());

    // Swap score to the back.
    std::iter_swap(begin(m_child_scores) + child,
                   end(m_child_scores) - 1);

    // Add the new node to expanded.
    m_expanded.emplace_back(std::make_unique<UCTNode>(
          m_child_scores.back().first, // move
          m_child_scores.back().second)); // score

    // Remove unexpanded node.
    m_child_scores.pop_back();
    return m_expanded.back().get();
}

void UCTNode::expand_all() {
    LOCK(get_mutex(), lock);

    // Expand all the child nodes.
    while (m_child_scores.size() > 0) {
        // Expand and pop last child.
        expand(m_child_scores.size() - 1);
    }
}

// Index of child with highest score.
size_t UCTNode::best_child() {
    assert(!m_child_scores.size().empty());

    auto best = size_t{0};
    auto bestScore = -1.0f;
    for (auto i = size_t{0}; i < m_child_scores.size(); i++) {
        auto score = m_child_scores[i].second;
        if (score > bestScore) {
            bestScore = score;
            best = i;
        }
    };
    return best;
}

UCTNode* UCTNode::uct_select_child(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_expanded.size() && !m_child_scores.empty());

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    for (const auto& child : m_expanded) {
        if (child->valid()) {
            parentvisits += child->get_visits();
            if (child->get_visits() > 0) {
                total_visited_policy += child->get_score();
            }
        }
    }

    if (parentvisits == 0) {
        // sort by score.
        parentvisits = 1;
    }

    auto numerator = std::sqrt((double)parentvisits);
    auto fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
    // Estimated eval for unknown nodes = original parent NN eval - reduction
    auto fpu_eval = get_net_eval(color) - fpu_reduction;

    // positive index mean expanded nodes, negative index for unexpanded.
    auto best_index = 0;
    auto best_value = -1000.0f;

    FastBoard board;
    board.reset_board(19);

    for (auto i = size_t{0}; i < m_expanded.size(); i++) {
        auto child = m_expanded[i].get();
        if (!child->valid()) continue;

        float winrate = fpu_eval;
        if (child->get_visits() > 0) {
            winrate = child->get_eval(color);
        }
        auto psa = child->get_score();
        auto denom = 1.0 + child->get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        auto value = winrate + puct;
        assert(value > -1000.0);

        if (value > best_value) {
            best_index = i;
            best_value = value;
        }
    }

    if (!m_child_scores.empty()) {
        auto best_unexpanded = best_child();

        // Unexpanded child
        auto winrate = fpu_eval;
        auto psa = m_child_scores[best_unexpanded].second;
        auto puct = cfg_puct * psa * numerator;
        auto value = winrate + puct;
        assert(value > -1000.0f);

        if (value > best_value) {
            return expand(best_unexpanded);
        }
    }

    return m_expanded[best_index].get();
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

void UCTNode::sort_children(int color) {
    LOCK(get_mutex(), lock);

    // Because sort depends on visits there's no need to expand all nodes.
    assert(!m_expanded.empty());
    std::stable_sort(rbegin(m_expanded), rend(m_expanded), NodeComp(color));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    LOCK(get_mutex(), lock);

    // Check if any of m_expanded have visits.
    auto any_visits = false;
    for (const auto& child : m_expanded) {
        if (child->get_visits()) {
            any_visits = true;
            break;
        }
    }

    if (any_visits == false && !m_child_scores.empty()) {
        // Expand best child for inclusion in search.
        expand(best_child());
    }

    assert(!m_expanded.empty());

    // NodeComp first checks visits so only need to consider m_expanded,
    return *(std::max_element(begin(m_expanded), end(m_expanded),
                              NodeComp(color))->get());
}

UCTNode* UCTNode::get_first_child() const {
    if (m_expanded.empty()) {
        return nullptr;
    }
    return m_expanded.front().get();
}

const std::vector<UCTNode::node_ptr_t>& UCTNode::get_children() const {
    return m_expanded;
}

size_t UCTNode::count_nodes() const {
    auto nodecount = m_expanded.size();
    for (const auto& child : m_expanded) {
        nodecount += child->count_nodes();
    }
    return nodecount;
}

size_t UCTNode::count_scores() const {
    auto count = m_child_scores.size();
    for (const auto& child : m_expanded) {
        count += child->count_scores();
    }
    return count;
}

// Used to find new root in UCTSearch
UCTNode::node_ptr_t UCTNode::find_child(const int move) {
    LOCK(get_mutex(), lock);

    if (m_has_children) {
        for (auto& child : m_expanded) {
            if (child->get_move() == move) {
                return std::move(child);
            }
        }
    }

    // Can happen if child is not expanded or move was resign.
    return nullptr;
}

UCTNode* UCTNode::get_nopass_child(FastState& state) {
    // Simplify logic by expanding all children. This method is only called
    // Rarely at root so avoid code complexity at small performance cost.
    expand_all();

    LOCK(get_mutex(), lock);

    for (const auto& child : m_expanded) {
        auto child_move = child->get_move();
        if (child_move != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), child_move)) {
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
