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
#include <limits.h>
#include <cmath>
#include <vector>
#include <utility>
#include <thread>
#include <algorithm>
#include <type_traits>

#include "FastBoard.h"
#include "UCTSearch.h"
#include "Timing.h"
#include "Random.h"
#include "Utils.h"
#include "Network.h"
#include "GTP.h"
#include "TTable.h"
#ifdef USE_OPENCL
#include "OpenCL.h"
#endif

using namespace Utils;

UCTSearch::UCTSearch(GameState & g)
    : m_rootstate(g),
      m_root(FastBoard::PASS, 0.0f),
      m_nodes(0),
      m_playouts(0),
      m_hasrunflag(false),
      m_runflag(nullptr) {
    set_playout_limit(cfg_max_playouts);
}

void UCTSearch::set_runflag(std::atomic<bool> * flag) {
    m_runflag = flag;
    m_hasrunflag = true;
}

SearchResult UCTSearch::play_simulation(GameState & currstate, UCTNode* const node) {
    const int color = currstate.get_to_move();
    const auto hash = currstate.board.get_hash();
    const auto komi = currstate.get_komi();

    bool has_updated = false;
    SearchResult result;

    TTable::get_TT()->sync(hash, komi, node);
    node->virtual_loss();

    if (!node->has_children() && m_nodes < MAX_TREE_SIZE) {
        bool success = node->create_children(m_nodes, currstate);
        if (success) {
            result = SearchResult(node->get_eval(color));
            has_updated = true;
        }
    }

    if (node->has_children() && !result.valid()) {
        UCTNode * next = node->uct_select_child(color);

        if (next != nullptr) {
            int move = next->get_move();

            if (move != FastBoard::PASS) {
                currstate.play_move(move);

                if (!currstate.superko()) {
                    result = play_simulation(currstate, next);
                } else {
                    next->invalidate();
                }
            } else {
                currstate.play_pass();
                result = play_simulation(currstate, next);
            }
        }
    }

    node->update(result.valid() && !has_updated, result.eval());
    TTable::get_TT()->update(hash, komi, node);

    return result;
}

void UCTSearch::dump_stats(KoState & state, UCTNode & parent) {
    const int color = state.get_to_move();

    if (!parent.has_children()) {
        return;
    }

    // sort children, put best move on top
    m_root.sort_root_children(color);

    UCTNode * bestnode = parent.get_first_child();

    if (bestnode->first_visit()) {
        return;
    }

    int movecount = 0;
    UCTNode * node = bestnode;

    while (node != nullptr) {
        if (++movecount > 2 && !node->get_visits()) break;

        std::string tmp = state.move_to_text(node->get_move());
        std::string pvstring(tmp);

        myprintf("%4s -> %7d (V: %5.2f%%) (N: %4.1f%%) PV: ",
            tmp.c_str(),
            node->get_visits(),
            node->get_visits() > 0 ? node->get_eval(color)*100.0f : 0.0f,
            node->get_score() * 100.0f);

        KoState tmpstate = state;

        tmpstate.play_move(node->get_move());
        pvstring += " " + get_pv(tmpstate, *node);

        myprintf("%s\n", pvstring.c_str());

        node = node->get_sibling();
    }
}

int UCTSearch::get_best_move(passflag_t passflag) {
    int color = m_rootstate.board.get_to_move();

    // make sure best is first
    m_root.sort_root_children(color);

    int bestmove = m_root.get_first_child()->get_move();

    // do we have statistics on the moves?
    if (m_root.get_first_child() != nullptr) {
        if (m_root.get_first_child()->first_visit()) {
            return bestmove;
        }
    }

    float bestscore = m_root.get_first_child()->get_eval(color);

    // do we want to fiddle with the best move because of the rule set?
     if (passflag & UCTSearch::NOPASS) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root.get_nopass_child();

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    bestscore = 1.0f;
                } else {
                    bestscore = nopass->get_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else {
        // Opponents last move was passing
        if (m_rootstate.get_last_move() == FastBoard::PASS) {
            // We didn't consider passing. Should we have and
            // end the game immediately?
            float score = m_rootstate.final_score();
            // do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses, I'll play on.\n");
            } else {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            }
        } else if (bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones
            float score = m_rootstate.final_score();
            // do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses :-(\n");
                // find a valid non-pass move
                UCTNode * nopass = m_root.get_nopass_child();
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        bestscore = 1.0f;
                    } else {
                        bestscore = nopass->get_eval(color);
                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else {
                myprintf("Passing wins :-)\n");
            }
        }
    }

    int visits = m_root.get_first_child()->get_visits();

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        // resigning allowed
        if ((passflag & UCTSearch::NORESIGN) == 0) {
            size_t movetresh = (m_rootstate.board.get_boardsize()
                                * m_rootstate.board.get_boardsize()) / 4;
            // bad score and visited enough
            if (bestscore < ((float)cfg_resignpct / 100.0f)
                && visits > 100
                && m_rootstate.m_movenum > movetresh) {
                myprintf("Score looks bad. Resigning.\n");
                bestmove = FastBoard::RESIGN;
            }
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(KoState & state, UCTNode & parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    // This breaks best probility = first in tree assumption
    parent.sort_root_children(state.get_to_move());

    LOCK(parent.get_mutex(), lock);
    UCTNode * bestchild = parent.get_first_child();
    int bestmove = bestchild->get_move();
    lock.unlock();

    std::string tmp = state.move_to_text(bestmove);

    std::string res(tmp);
    res.append(" ");

    state.play_move(bestmove);

    std::string next = get_pv(state, *bestchild);
    res.append(next);

    // Resort according to move probability
    lock.lock();
    parent.sort_children();

    return res;
}

void UCTSearch::dump_analysis(int playouts) {
    GameState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    std::string pvstring = get_pv(tempstate, m_root);
    float winrate = 100.0f * m_root.get_eval(color);
    myprintf("Playouts: %d, Win: %5.2f%%, PV: %s\n",
             playouts, winrate, pvstring.c_str());
}

bool UCTSearch::is_running() {
    return m_run;
}

bool UCTSearch::playout_limit_reached() {
    return m_playouts >= m_maxplayouts;
}

void UCTWorker::operator()() {
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        m_search->play_simulation(*currstate, m_root);
        m_search->increment_playouts();
    } while(m_search->is_running() && !m_search->playout_limit_reached());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

int UCTSearch::think(int color, passflag_t passflag) {
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set side to move
    m_rootstate.board.set_to_move(color);

    // set up timing info
    Time start;

    m_rootstate.get_timecontrol().set_boardsize(m_rootstate.board.get_boardsize());
    auto time_for_move = m_rootstate.get_timecontrol().max_time_for_move(color);

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list off legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root.virtual_loss();
    m_root.create_children(m_nodes, m_rootstate);
    m_root.kill_superkos(m_rootstate);
    if (cfg_noise) {
        m_root.dirichlet_noise(0.25f, 0.03f);
    }

    myprintf("NN eval=%f\n", m_root.get_eval(color));

    m_run = true;
    m_playouts = 0;

    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, &m_root));
    }

    bool keeprunning = true;
    int last_update = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        play_simulation(*currstate, &m_root);
        increment_playouts();

        Time elapsed;
        int centiseconds_elapsed = Time::timediff(start, elapsed);

        // output some stats every few seconds
        // check if we should still search
        if (centiseconds_elapsed - last_update > 250) {
            last_update = centiseconds_elapsed;
            dump_analysis(static_cast<int>(m_playouts));
        }
        keeprunning = (centiseconds_elapsed < time_for_move
                        && (!m_hasrunflag || (*m_runflag)));
        keeprunning &= !playout_limit_reached();
    } while(keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();
    if (!m_root.has_children()) {
        return FastBoard::PASS;
    }
    m_rootstate.stop_clock(color);

    // display search info
    myprintf("\n");

    dump_stats(m_rootstate, m_root);

    Time elapsed;
    int centiseconds_elapsed = Time::timediff(start, elapsed);
    if (centiseconds_elapsed > 0) {
        myprintf("%d visits, %d nodes, %d playouts, %d n/s\n\n",
                 m_root.get_visits(),
                 static_cast<int>(m_nodes),
                 static_cast<int>(m_playouts),
                 (m_playouts * 100) / (centiseconds_elapsed+1));
    }
    int bestmove = get_best_move(passflag);
    return bestmove;
}

void UCTSearch::ponder() {
    m_run = true;
    m_playouts = 0;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, &m_root));
    }
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        play_simulation(*currstate, &m_root);
        increment_playouts();
    } while(!Utils::input_pending() && (!m_hasrunflag || (*m_runflag)));

    // stop the search
    m_run = false;
    tg.wait_all();
    // display search info
    myprintf("\n");
    dump_stats(m_rootstate, m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root.get_visits(), (int)m_nodes);
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    if (playouts == 0) {
        m_maxplayouts = std::numeric_limits<decltype(m_maxplayouts)>::max();
    } else {
        m_maxplayouts = playouts;
    }
}
