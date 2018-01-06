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

#ifndef TTABLE_H_INCLUDED
#define TTABLE_H_INCLUDED

#include "config.h"

#include <vector>

#include "SMP.h"
#include "UCTNode.h"

class TTEntry {
public:
    TTEntry() = default;

    std::uint64_t m_hash{0};
    int m_visits;
    double m_eval_sum;
};

class TTable {
public:
    /*
        return the global TT
    */
    static TTable& get_TT(void);

    /*
        update corresponding entry
    */
    void update(std::uint64_t hash, const float komi, const UCTNode * node);

    /*
        sync given node with TT
    */
    void sync(std::uint64_t hash, const float komi, UCTNode * node);

private:
    TTable(int size = 500000);

    SMP::Mutex m_mutex;
    std::vector<TTEntry> m_buckets;
    float m_komi;
};

#endif
