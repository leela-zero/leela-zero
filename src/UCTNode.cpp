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

UCTEdge::UCTEdge(int vertex, float score)
    : m_move(vertex), m_score(score), m_child(nullptr) {
}

int UCTEdge::get_move() const {
    return m_move;
}

float UCTEdge::get_score() const {
    return m_score;
}

void UCTEdge::set_score(float score) {
    m_score = score;
}

UCTNode* UCTEdge::get_child() const {
    return m_child.get();
}

void UCTEdge::invalidate() {
    m_valid = false;
}

bool UCTEdge::valid() const {
    return m_valid;
}

int UCTEdge::get_visits() const {
    if (m_child) {
        return m_child->get_visits();
    }
    return 0;
}

float UCTEdge::get_eval(int tomove) const {
    if (m_child) {
        return m_child->get_eval(tomove);
    }
    return 0.0f;
}

void UCTEdge::create_child(std::atomic<int>& nodecount) {
    if (m_child) return;

    LOCK(m_nodemutex, lock);
    // Check whether another thread beat us to it (after taking the lock).
    if (m_child) return;
    m_child = std::make_unique<UCTNode>(m_move, m_score);
    lock.unlock();

    nodecount++;
}

UCTNode* UCTEdge::get_or_create_child(std::atomic<int>& nodecount) {
    create_child(nodecount);
    return m_child.get();
}

UCTEdge::node_ptr_t UCTEdge::pass_or_create_child(std::atomic<int>& nodecount) {
    create_child(nodecount);
    return std::move(m_child);
}

bool UCTEdge::first_visit() const {
    return !m_child;
}

UCTNode::UCTNode(int vertex, float score) : m_move(vertex), m_score(score) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_edges(std::atomic<int>& edgecount,
                           GameState& state,
                           float& eval) {
    // check whether somebody beat us to it (atomic)
    if (has_edges()) {
        return false;
    }
    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }
    // check whether somebody beat us to it (after taking the lock)
    if (has_edges()) {
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

    // If the sum is 0 or a denormal, then don't try to normalize.
    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    }

    link_nodelist(edgecount, nodelist);
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& edgecount,
                            std::vector<Network::scored_node>& nodelist) {
    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(get_mutex(), lock);

    m_edges.reserve(nodelist.size());
    for (const auto& node : nodelist) {
        m_edges.emplace_back(
            std::make_unique<UCTEdge>(node.second, node.first)
        );
    }

    edgecount += m_edges.size();
    m_has_edges = true;
}

void UCTNode::kill_superkos(const KoState& state) {
    for (auto& edge : m_edges) {
        auto move = edge->get_move();
        if (move != FastBoard::PASS) {
            KoState mystate = state;
            mystate.play_move(move);

            if (mystate.superko()) {
                // Don't delete edges for now, just mark them invalid.
                edge->invalidate();
            }
        }
    }

    // Now do the actual deletion.
    m_edges.erase(
        std::remove_if(begin(m_edges), end(m_edges),
                       [](const auto &edge) { return !edge->valid(); }),
        end(m_edges)
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
    auto edge_cnt = m_edges.size();

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < edge_cnt; i++) {
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

    edge_cnt = 0;
    for (auto& edge : m_edges) {
        auto score = edge->get_score();
        auto eta_a = dirichlet_vector[edge_cnt++];
        score = score * (1 - epsilon) + epsilon * eta_a;
        edge->set_score(score);
    }
}

void UCTNode::randomize_first_proportionally() {
    auto accum = std::uint64_t{0};
    auto accum_vector = std::vector<decltype(accum)>{};
    for (const auto& edge : m_edges) {
        accum += edge->get_visits();
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

    assert(m_edges.size() >= index);

    // Now swap the edge at index with the first edge
    std::iter_swap(begin(m_edges), begin(m_edges) + index);
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

bool UCTNode::has_edges() const {
    return m_has_edges;
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

UCTEdge* UCTNode::uct_select_edge(int color) {
    UCTEdge* best = nullptr;
    auto best_value = -1000.0f;

    LOCK(get_mutex(), lock);

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    for (const auto& edge : m_edges) {
        if (edge->valid()) {
            auto visits = edge->get_visits();
            parentvisits += visits;
            if (visits > 0) {
                total_visited_policy += edge->get_score();
            }
        }
    }

    auto numerator = static_cast<float>(std::sqrt((double)parentvisits));
    auto fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
    // Estimated eval for unknown nodes = original parent NN eval - reduction
    auto fpu_eval = get_net_eval(color) - fpu_reduction;

    for (const auto& edge : m_edges) {
        if (!edge->valid()) {
            continue;
        }

        float winrate = fpu_eval;
        if (edge->get_visits() > 0) {
            winrate = edge->get_eval(color);
        }
        auto psa = edge->get_score();
        auto denom = 1.0f + edge->get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        auto value = winrate + puct;
        assert(value > -1000.0f);

        if (value > best_value) {
            best_value = value;
            best = edge.get();
        }
    }

    assert(best != nullptr);
    return best;
}

class EdgeComp : public std::binary_function<UCTNode::edge_ptr_t&,
                                             UCTNode::edge_ptr_t&, bool> {
public:
    EdgeComp(int color) : m_color(color) {};
    bool operator()(const UCTNode::edge_ptr_t& a,
                    const UCTNode::edge_ptr_t& b) {
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

void UCTNode::sort_edges(int color) {
    LOCK(get_mutex(), lock);
    std::stable_sort(rbegin(m_edges), rend(m_edges), EdgeComp(color));
}

UCTEdge* UCTNode::get_best_root_edge(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_edges.empty());

    return std::max_element(begin(m_edges), end(m_edges),
                            EdgeComp(color))->get();
}

UCTEdge* UCTNode::get_first_edge() const {
    if (m_edges.empty()) {
        return nullptr;
    }
    return m_edges.front().get();
}

const std::vector<UCTNode::edge_ptr_t>& UCTNode::get_edges() const {
    return m_edges;
}

size_t UCTNode::count_nodes() const {
    auto nodecount = size_t{1};
    if (m_has_edges) {
        for (auto& edge : m_edges) {
            auto child = edge->get_child();
            if (child) {
                nodecount += child->count_edges();
            }
        }
    }
    return nodecount;
}

size_t UCTNode::count_edges() const {
    auto edgecount = size_t{0};
    if (m_has_edges) {
        edgecount += m_edges.size();
        for (auto& edge : m_edges) {
            auto child = edge->get_child();
            if (child) {
                edgecount += child->count_edges();
            }
        }
    }
    return edgecount;
}

// Used to find new root in UCTSearch
UCTEdge* UCTNode::find_edge(const int move) {
    if (m_has_edges) {
        for (auto& edge : m_edges) {
            if (edge->get_move() == move) {
                return edge.get();
            }
        }
    }

    // Can happen if we resigned or edges are not expanded
    return nullptr;
}

UCTEdge* UCTNode::get_nopass_edge(FastState& state) const {
    for (const auto& edge : m_edges) {
        /* If we prevent the engine from passing, we must bail out when
           we only have unreasonable moves to pick, like filling eyes.
           Note that this knowledge isn't required by the engine,
           we require it because we're overruling its moves. */
        if (edge->get_move() != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), edge->get_move())) {
            return edge.get();
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
