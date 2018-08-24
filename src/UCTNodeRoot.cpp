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
//#include <execution>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "UCTSearch.h"
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
        auto policy = child->get_policy();
        auto eta_a = dirichlet_vector[child_cnt++];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
        child->set_policy(policy);
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

float white_net_eval(Network & net, GameState& root_state) {
    float net_eval;
    if (cfg_use_symmetries) {
        if (cfg_fixed_symmetry == -1) {
            net_eval = net.get_output(&root_state, Network::Ensemble::AVERAGE, 8, true).winrate;
        }
        else {
            net_eval = net.get_output(&root_state, Network::Ensemble::DIRECT, cfg_fixed_symmetry, true).winrate;
        }
    }
    else {
        net_eval = net.get_output(&root_state, Network::Ensemble::RANDOM_SYMMETRY).winrate;
    }
    if (root_state.get_to_move() == FastBoard::WHITE) {
        return net_eval;
    }
    else {
        return 1.0 - net_eval;
    }
}

float inv_wr(float wr) {
    return -log(1.0 / wr - 1.0) / 2.0;
}

void binary_search_komi(GameState& root_state, float shift, float high, float low, float target_wr, int steps, std::function<float(float)> get_white_eval) {
    while (steps-- > 0) {
        root_state.m_stm_komi = (high + low) / 2.0;
        auto net_eval = get_white_eval(root_state.m_stm_komi);
        if (inv_wr(net_eval) + shift > inv_wr(target_wr)) {
            high = root_state.m_stm_komi;
        }
        else {
            low = root_state.m_stm_komi;
        }
    }
    root_state.m_stm_komi = (high + low) / 2.0;
}

void adjust_up_komi(GameState& root_state, float shift, float target_wr, std::function<float(float)> get_white_eval) {
	float net_eval;
    float old_komi;
    float orig_komi = root_state.m_stm_komi;
    int steps = cfg_steps;
	do {
        old_komi = root_state.m_stm_komi;
        if (old_komi < -7.5) {
            root_state.m_stm_komi = -7.5;
        }
        else if (old_komi < 7.5) {
            root_state.m_stm_komi = 7.5;
        }
        else {
            root_state.m_stm_komi = 2.0f * old_komi;
            if (cfg_orig_policy && root_state.eval_invalid()) {
                root_state.m_stm_komi = 7.5;
                return;
            }
        }        
        if (steps-- < 0) { root_state.m_stm_komi = orig_komi; return; }
        net_eval = get_white_eval(root_state.m_stm_komi);
        if (inv_wr(net_eval) + shift > inv_wr(cfg_min_wr + cfg_wr_margin) && (root_state.m_stm_komi == 7.5 || root_state.m_stm_komi == -7.5 || root_state.m_stm_komi == cfg_target_komi)) { return; }
	} while (inv_wr(net_eval) + shift < inv_wr(target_wr));
    binary_search_komi(root_state, shift, root_state.m_stm_komi, old_komi, target_wr, cfg_steps, get_white_eval);
}

void adjust_down_komi(GameState& root_state, float shift, float target_wr, std::function<float(float)> get_white_eval) {
    float net_eval;
    float old_komi;
    float orig_komi = root_state.m_stm_komi;
    int steps = cfg_steps;
    if (cfg_nonslack) {
        do {
            old_komi = root_state.m_stm_komi;
            if (old_komi > 7.5) {
                root_state.m_stm_komi = 7.5;
            }
            else if (old_komi > -7.5) {
                root_state.m_stm_komi = -7.5;
            }
            else {
                root_state.m_stm_komi = 2.0f * old_komi;
                if (cfg_orig_policy && root_state.eval_invalid()) {
                    root_state.m_stm_komi = -7.5;
                    return;
                }
            }
            if (steps-- < 0) { root_state.m_stm_komi = orig_komi; return; }
            net_eval = get_white_eval(root_state.m_stm_komi);
            if (inv_wr(net_eval) + shift < inv_wr(cfg_max_wr - cfg_wr_margin) && (root_state.m_stm_komi == 7.5 || root_state.m_stm_komi == -7.5 || root_state.m_stm_komi == cfg_target_komi)) { return; }
        } while (inv_wr(net_eval) + shift > inv_wr(target_wr));
        binary_search_komi(root_state, shift, old_komi, root_state.m_stm_komi, target_wr, cfg_steps, get_white_eval);
    }
    else {
        old_komi = root_state.m_stm_komi;
        root_state.m_stm_komi = cfg_target_komi;
        net_eval = get_white_eval(root_state.m_stm_komi);
        if (inv_wr(net_eval) + shift < inv_wr(target_wr)) {
            binary_search_komi(root_state, shift, old_komi, root_state.m_stm_komi, target_wr, cfg_steps, get_white_eval);
        }
    }
}

float adjust_komi(GameState& root_state, float root_eval, float target_wr, bool opp, std::function<float(float)> get_white_eval) {
    float shift;
    if (root_eval > cfg_max_wr || root_eval < cfg_min_wr) {
        auto net_eval = get_white_eval(root_state.m_stm_komi);
        if (cfg_noshift) {
            shift = 0.0f;
            root_eval = net_eval;
        }
        else {
            shift = inv_wr(root_eval) - inv_wr(net_eval);
        }
        Utils::myprintf("%f, %f, %f\n", root_eval, net_eval, shift); //
        if (opp && !cfg_pos && !cfg_neg) {
            if (root_eval < target_wr) {
                adjust_up_komi(root_state, shift, target_wr, get_white_eval);
                return -1.0f;
            }
            else if (root_eval > target_wr) {
                adjust_down_komi(root_state, shift, target_wr, get_white_eval);
                return -1.0f;
            }
        }
        else {
            if (root_eval < cfg_min_wr) {
                target_wr = cfg_min_wr + cfg_wr_margin;
                adjust_up_komi(root_state, shift, target_wr, get_white_eval);
                return target_wr;
            }
            else if ((root_state.m_stm_komi != cfg_target_komi || cfg_nonslack) && root_eval > cfg_max_wr) {
                target_wr = cfg_max_wr - cfg_wr_margin;
                adjust_down_komi(root_state, shift, target_wr, get_white_eval);
                return target_wr;
            }
        }
    }
    if (cfg_nonslack) {
        float net_eval;
        if (cfg_noshift) { shift = 0.0; }
        else {
            net_eval = get_white_eval(root_state.m_stm_komi);
            shift = inv_wr(root_eval) - inv_wr(net_eval);
        }
        auto komi = root_state.m_stm_komi;
        root_state.m_stm_komi = cfg_target_komi;
        net_eval = get_white_eval(root_state.m_stm_komi);
        if (inv_wr(net_eval) + shift < inv_wr(cfg_min_wr + cfg_wr_margin) || inv_wr(net_eval) + shift > inv_wr(cfg_max_wr - cfg_wr_margin)) {
            root_state.m_stm_komi = komi;
        }
    }
}

float mean_white_eval(Network & net, std::vector<std::shared_ptr<Sym_State>>& ssi, float komi) {
    auto num_positions = ssi.size();
    // use threads = cfg_num_threads, parallelism?
    if (ssi[0]->state.m_stm_komi != komi) {
        /*
        Concurrency::parallel_for()
        std::for_each(std::execution::par_unseq, ssi.begin(), ssi.end(),
            [komi](const std::shared_ptr<Sym_State> &sym_state) {sym_state->state.m_stm_komi = komi;
        sym_state->winrate = Network::get_scored_moves(&sym_state->state, Network::Ensemble::DIRECT, sym_state->symmetry, true).winrate; });
        */
        for (auto j = 0; j < num_positions; j++) {
            ssi[j]->state.m_stm_komi = komi;
            ssi[j]->winrate = net.get_output(&ssi[j]->state, Network::Ensemble::DIRECT, ssi[j]->symmetry, true).winrate;
        }
    }
    auto mean = accumulate(ssi.begin(), ssi.end(), 0.0f, [](float s, std::shared_ptr<Sym_State> sym_state) {return s + sym_state->winrate; }) / num_positions;
    if (ssi[0]->state.get_to_move() == FastBoard::BLACK) { return 1.0f - mean; }
    else { return mean; }
}

void UCTNode::clear(Network & net, std::atomic<int>& nodes, GameState& root_state, float& eval) {
    net.clear_cache();
    m_visits = 0;
    m_blackevals = 0.0;
    m_min_psa_ratio_children = 2.0;
    m_children.clear();

    create_children(net, nodes, root_state, eval);
    inflate_all_children();
    kill_superkos(root_state);
    if (root_state.eval_invalid()) {
        GameState tmpstate = root_state;
        tmpstate.play_move(get_first_child()->get_move());
        eval = 1.0f - white_net_eval(net, tmpstate);
    }
    update(eval);
    eval = (root_state.get_to_move() == FastBoard::BLACK ? eval : 1.0f - eval);
}

void UCTNode::prepare_root_node(Network & network, int color, // redundant argument?
                                std::atomic<int>& nodes,
                                GameState& root_state, UCTSearch * search) {
    float root_eval;
    const auto had_children = has_children();
    if (expandable()) {
        create_children(network, nodes, root_state, root_eval);
    }

    // There are a lot of special cases where code assumes
    // all children of the root are inflated, so do that.
    inflate_all_children();

    // Remove illegal moves, so the root move list is correct.
    // This also removes a lot of special cases.
    kill_superkos(root_state);

    if (had_children) {
        root_eval = get_eval(color);
    }
    else {
        if (root_state.eval_invalid()) {
            GameState tmpstate = root_state;
            tmpstate.play_move(get_first_child()->get_move());
            root_eval = 1.0f - white_net_eval(network, tmpstate);
        }
        update(root_eval);
        root_eval = (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval);
    }

    std::array<std::vector<std::shared_ptr<Sym_State>>, 2> ss;
    ss[0].reserve(cfg_adj_positions + cfg_num_threads);
    ss[1].reserve(cfg_adj_positions + cfg_num_threads);
    for (auto i = 0; i < 2; i++) {
        for (auto j = 0; j < cfg_num_threads; j++) {
            ss[i].insert(ss[i].end(), search->sym_states[i][j].begin(),search->sym_states[i][j].end());
        }
    }

    Utils::myprintf("black positions: %d\n", ss[0].size());
    Utils::myprintf("white positions: %d\n", ss[1].size());

    if (cfg_dyn_komi) {
        if (cfg_collect_during_search) {
            auto hash = root_state.board.get_ko_hash();
            auto index = root_state.m_ko_hash_history.size() - 1;
            for (auto i = 0; i < 2; i++) {
                auto num_removed = 0;
                auto colored_root_eval = (i == color ? root_eval : 1.0f - root_eval);
                if (!cfg_use_root_for_diff) { colored_root_eval = 0.0f; }
                for (auto j = 0; j < ss[i].size(); j++) {
                    bool to_remove = false;
                    if ((!search->collecting && // root changed, clear all sym_states that are not descendents of root
                        (ss[i][j]->state.m_ko_hash_history.size() <= index
                            || ss[i][j]->state.m_ko_hash_history[index] != hash))) {
                        num_removed++;
                    }
                    else {
                        ss[i][j]->diff = abs(ss[i][j]->winrate - colored_root_eval);
                        ss[i][j - num_removed] = ss[i][j];
                        if (!cfg_use_root_for_diff) { colored_root_eval += ss[i][j]->winrate; }
                    }
                }
                ss[i].resize(ss[i].size() - num_removed);
                if (!cfg_use_root_for_diff) {
                    colored_root_eval /= (ss[i].size());
                    for (auto j = 0; j < ss[i].size(); j++) { ss[i][j]->diff = abs(ss[i][j]->winrate - colored_root_eval); }
                }
            }
            Utils::myprintf("deleting non-descendents\n");
            Utils::myprintf("black positions: %d\n", ss[0].size());
            Utils::myprintf("white positions: %d\n", ss[1].size());
        }

        bool to_adjust = false;
        // no need to collect ss[0] or ss[1] if cfg_pos or cfg_neg ..
        if (search->collecting || (ss[0].size() >= cfg_adj_positions && ss[1].size() >= cfg_adj_positions)) {
            to_adjust = true;
            auto num_positions = ceil(cfg_adj_positions * cfg_adj_pct / 100.0);
            for (auto i = 0; i < 2; i++) {
                std::nth_element(ss[i].begin(), ss[i].begin() + num_positions, ss[i].end(), 
                    [](std::shared_ptr<Sym_State>& sym_state1, std::shared_ptr<Sym_State>& sym_state2) {
                    return sym_state1->diff < sym_state2->diff; });
                ss[i].resize(num_positions);
            }
            Utils::myprintf("keeping %f%% closest evals\n", cfg_adj_pct);
            Utils::myprintf("black positions: %d\n", ss[0].size());
            Utils::myprintf("white positions: %d\n", ss[1].size());
        }

        if (!cfg_collect_during_search || to_adjust) {
            std::function<float(float)> get_white_eval;
            if (cfg_collect_during_search) {
                get_white_eval = [&network, &ss, color](float komi_) {return mean_white_eval(network, ss[color], komi_); };
            }
            else {
                get_white_eval = [&network, &root_state](float) {return white_net_eval(network, root_state); };
            }
            auto white_root_eval = get_raw_eval(FastBoard::WHITE);
            auto komi = root_state.m_stm_komi;
            auto opp_komi = root_state.m_opp_komi;
            auto tmp_komi = komi;
            auto target_wr = adjust_komi(root_state, white_root_eval, -1.0f, false, get_white_eval);
            tmp_komi = root_state.m_stm_komi;
            if (komi != root_state.m_stm_komi) {
                clear(network, nodes, root_state, root_eval);
                //search->sym_states[color].assign(cfg_num_threads, {});
            }
            if (komi != root_state.m_stm_komi || cfg_pos || cfg_neg) {
                if (!(cfg_pos || cfg_neg) && (root_state.m_stm_komi == cfg_target_komi || root_state.m_stm_komi == 7.5 || root_state.m_stm_komi == -7.5)) {
                    root_state.m_opp_komi = root_state.m_stm_komi;
                }
                else {
                    GameState tmpstate = root_state;
                    tmpstate.play_move(get_first_child()->get_move());
                    if (cfg_collect_during_search) {
                        get_white_eval = [&network, &ss, color](float komi_) {return mean_white_eval(network, ss[!color], komi_); };
                    }
                    else {
                        get_white_eval = [&network, &tmpstate](float) {return white_net_eval(network, tmpstate); };
                    }
                    adjust_komi(tmpstate, white_root_eval, target_wr, true, get_white_eval);
                    root_state.m_opp_komi = tmpstate.m_stm_komi;
                }

                if (opp_komi != root_state.m_opp_komi) {
                    clear(network, nodes, root_state, root_eval);
                    //search->sym_states[!color].assign(cfg_num_threads, {});
                }
                else if (!(cfg_pos || cfg_neg)) {
                    root_state.m_stm_komi = komi;
                }
            }
            if (!(cfg_pos || cfg_neg) && (root_state.m_opp_komi == cfg_target_komi || root_state.m_opp_komi == 7.5 || root_state.m_opp_komi == -7.5)) {
                root_state.m_stm_komi = root_state.m_opp_komi;
            }
            if (tmp_komi != root_state.m_stm_komi) {
                clear(network, nodes, root_state, root_eval);
                //search->sym_states[color].assign(cfg_num_threads, {});
            }
        }
        search->sym_states[0].assign(cfg_num_threads, {});
        search->sym_states[1].assign(cfg_num_threads, {});
        Utils::myprintf("NN eval=%f\n", root_eval);
        Utils::myprintf("komi=%f\n", root_state.m_stm_komi);
        Utils::myprintf("opp_komi=%f\n", root_state.m_opp_komi);
    }
    else {
        Utils::myprintf("NN eval=%f\n", root_eval);
    }
    search->collecting = false;

    if (cfg_noise) {
        // Adjust the Dirichlet noise's alpha constant to the board size
        auto alpha = 0.03f * 361.0f / BOARD_SQUARES;
        dirichlet_noise(0.25f, alpha);
    }
}
