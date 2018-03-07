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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "KoState.h"
#include "Random.h"
#include "UCTNode.h"

/*
 * These functions belong to UCTNode but should only be called on the root node
 * of UCTSearch and have been seperated to increase code clarity.
 */

UCTNode* UCTNode::get_first_child() const {
    if (m_children.empty()) {
        return nullptr;
    }

    return m_children.front().get();
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
    auto accum = std::uint64_t{0};
    auto accum_vector = std::vector<decltype(accum)>{};
    for (const auto& child : m_children) {
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

    assert(m_children.size() >= index);

    // Now swap the child at index with the first child
    std::iter_swap(begin(m_children), begin(m_children) + index);
}

UCTNode* UCTNode::get_nopass_child(FastState& state) const {
    for (const auto& child : m_children) {
        /* If we prevent the engine from passing, we must bail out when
           we only have unreasonable moves to pick, like filling eyes.
           Note that this knowledge isn't required by the engine,
           we require it because we're overruling its moves. */
        if (child->m_move != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), child->m_move)) {
            return child.get();
        }
    }
    return nullptr;
}

// Used to find new root in UCTSearch
UCTNode::node_ptr_t UCTNode::find_child(const int move) {
    if (m_has_children) {
        for (auto& child : m_children) {
            if (child->get_move() == move) {
                return std::move(child);
            }
        }
    }

    // Can happen if we resigned or children are not expanded
    return nullptr;
}
