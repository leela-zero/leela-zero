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

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <memory>
#include <atomic>
#include <tuple>

#include "GameState.h"
#include "UCTNode.h"

class SearchResult {
public:
    SearchResult() = default;
    explicit SearchResult(float e)
        : m_valid(true), m_eval(e) {};
    bool valid() { return m_valid;  }
    float eval() { return m_eval;  }
private:
    bool m_valid{false};
    float m_eval{0.0f};
};

class UCTSearch {
public:
    /*
        Depending on rule set and state of the game, we might
        prefer to pass, or we might prefer not to pass unless
        it's the last resort. Same for resigning.
    */
    using passflag_t = int;
    static constexpr passflag_t NORMAL   = 0;
    static constexpr passflag_t NOPASS   = 1 << 0;
    static constexpr passflag_t NORESIGN = 1 << 1;

    /*
        Maximum size of the tree in memory.
    */
    static constexpr int MAX_TREE_SIZE = 10000000;

    UCTSearch(GameState & g);
    int think(int color, passflag_t passflag = NORMAL);
    void set_playout_limit(int playouts);
    void set_runflag(std::atomic<bool> * flag);
    void set_analyzing(bool flag);
    void set_quiet(bool flag);
    void ponder();
    bool is_running();
    bool playout_limit_reached();
    void increment_playouts();
    SearchResult play_simulation(GameState & currstate, UCTNode * const node);

private:
    void dump_stats(KoState & state, UCTNode & parent);
    std::string get_pv(KoState & state, UCTNode & parent);
    void dump_analysis(int playouts);
    int get_best_move(passflag_t passflag);

    GameState & m_rootstate;
    UCTNode m_root;
    std::atomic<int> m_nodes;
    std::atomic<int> m_playouts;
    std::atomic<bool> m_run;
    int m_maxplayouts;

    // For external control
    bool m_hasrunflag;
    std::atomic<bool> * m_runflag;
};

class UCTWorker {
public:
    UCTWorker(GameState & state, UCTSearch * search, UCTNode * root)
      : m_rootstate(state), m_search(search), m_root(root) {};
    void operator()();
private:
    GameState & m_rootstate;
    UCTSearch * m_search;
    UCTNode * m_root;
};

#endif
