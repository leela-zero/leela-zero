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
#include "NNCache.h"
#include "Random.h"
#include "UCTNode.h"
#include "Utils.h"
#include "GTP.h"

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

    for (auto& v : dirichlet_vector) {
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
    auto accum = 0.0;
    auto norm_factor = 0.0;
    auto accum_vector = std::vector<double>{};

    for (const auto& child : m_children) {
        auto visits = child->get_visits();
        if (norm_factor == 0.0) {
            norm_factor = visits;
            // Nonsensical options? End of game?
            if (visits <= cfg_random_min_visits) {
                return;
            }
        }
        if (visits > cfg_random_min_visits) {
            accum += std::pow(visits / norm_factor,
                              1.0 / cfg_random_temp);
            accum_vector.emplace_back(accum);
        }
    }

    auto distribution = std::uniform_real_distribution<double>{0.0, accum};
    auto pick = distribution(Random::get_Rng());
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

    assert(m_children.size() > index);

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

// Used to find new root in UCTSearch.
std::unique_ptr<UCTNode> UCTNode::find_child(const int move) {
    for (auto& child : m_children) {
        if (child.get_move() == move) {
             // no guarantee that this is a non-inflated node
            child.inflate();
            return std::unique_ptr<UCTNode>(child.release());
        }
    }

    // Can happen if we resigned or children are not expanded
    return nullptr;
}

void UCTNode::inflate_all_children() {
    for (const auto& node : get_children()) {
        node.inflate();
    }
}


const int cfg_steps = 8;
const float target_komi = 7.5f;

float white_net_eval(GameState root_state) {
    auto net_eval = Network::get_scored_moves(&root_state, Network::Ensemble::AVERAGE, 8, true).winrate;
    if (root_state.get_to_move() == FastBoard::WHITE) {
        return net_eval;
    }
    else {
        return 1.0 - net_eval;
    }
}

void binary_search_komi(GameState& root_state, float factor, float high, float low, int steps) {
    while (steps-- > 0) {
        root_state.m_komi = (high + low) / 2.0;
        auto net_eval = white_net_eval(root_state);
        if (net_eval * factor > cfg_mid_wr) {
            high = root_state.m_komi;
        }
        else {
            low = root_state.m_komi;
        }
    }
    root_state.m_komi = low;
}

void adjust_up_komi(GameState& root_state, float factor) {
	float net_eval;
	do {
		root_state.m_komi = 2.0f * root_state.m_komi - (target_komi - 7.5f);
        net_eval = white_net_eval(root_state);
	} while (net_eval * factor < cfg_mid_wr);
	binary_search_komi(root_state, factor, root_state.m_komi, (root_state.m_komi + target_komi - 7.5f) / 2.0f, cfg_steps);
}

void adjust_down_komi(GameState& root_state, float factor) {
	auto komi = root_state.m_komi;
	root_state.m_komi = target_komi;
	auto net_eval = white_net_eval(root_state);
	if (net_eval * factor < cfg_mid_wr) {
		binary_search_komi(root_state, factor, komi, target_komi, cfg_steps);
	}
}

void adjust_komi(GameState& root_state, bool opp) { //, float root_eval) {
    auto root_eval = white_net_eval(root_state);
    if (opp) {
        if (root_eval < cfg_mid_wr) {
            adjust_up_komi(root_state, 1.0f);
        }
        else if (root_eval > cfg_max_wr) {
            adjust_down_komi(root_state, 1.0f);
        }
    }
    else {
        if (root_eval < cfg_min_wr) {
            //auto net_eval = white_net_eval(root_state);
            adjust_up_komi(root_state, 1.0f); // root_eval / net_eval);
        }
        else if (root_state.m_komi != target_komi && root_eval > cfg_max_wr) {
            //auto net_eval = white_net_eval(root_state);
            adjust_down_komi(root_state, 1.0f); // root_eval / net_eval);
        }
    }
}

void UCTNode::prepare_root_node(int color,
                                std::atomic<int>& nodes,
                                GameState& root_state) {
    float root_eval;
    const auto had_children = has_children();
    if (expandable()) {
        create_children(nodes, root_state, root_eval);
    }
    if (had_children) {
        root_eval = get_eval(color);
    } else {
        update(root_eval);
        root_eval = (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval);
    }
    auto komi = root_state.m_komi;
    adjust_komi(root_state, false);
    if (komi != root_state.m_komi) {
        NNCache::get_NNCache().clear_cache();
        m_visits = 0;
        m_blackevals = 0.0;
        m_min_psa_ratio_children = 2.0;
        m_children.clear();
        create_children(nodes, root_state, root_eval);
        root_eval = (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval);
    }

    // There are a lot of special cases where code assumes
    // all children of the root are inflated, so do that.
    inflate_all_children();

    // Remove illegal moves, so the root move list is correct.
    // This also removes a lot of special cases.
    kill_superkos(root_state);

    if (komi != root_state.m_komi) {
        komi = root_state.m_opp_komi;
        if (root_state.m_komi == target_komi) {
            root_state.m_opp_komi = target_komi;
        }
        else {
            GameState tmpstate = root_state;
            tmpstate.play_move(get_first_child()->get_move());
            adjust_komi(tmpstate, true);
            root_state.m_opp_komi = tmpstate.m_komi;
        }
        if (komi != root_state.m_opp_komi) {
            NNCache::get_NNCache().clear_cache();
        }
    }
    if (root_state.m_opp_komi == target_komi) {
        root_state.m_komi = target_komi;
    }

    Utils::myprintf("NN eval=%f\n", root_eval);
    Utils::myprintf("komi=%f\n", root_state.m_komi);
    Utils::myprintf("opp_komi=%f\n", root_state.m_opp_komi);

    if (cfg_noise) {
        // Adjust the Dirichlet noise's alpha constant to the board size
        auto alpha = 0.03f * 361.0f / BOARD_SQUARES;
        dirichlet_noise(0.25f, alpha);
    }
}
