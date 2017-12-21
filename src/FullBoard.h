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

#ifndef FULLBOARD_H_INCLUDED
#define FULLBOARD_H_INCLUDED

#include "config.h"
#include <cstdint>
#include "FastBoard.h"

class FullBoard : public FastBoard {
public:
    int remove_string(int i);
    int update_board(const int color, const int i);

    std::uint64_t calc_hash(void);
    std::uint64_t calc_ko_hash(void);
    std::uint64_t get_hash(void) const;
    std::uint64_t get_ko_hash(void) const;

    void reset_board(int size);
    void display_board(int lastmove = -1);

    std::uint64_t m_hash;
    std::uint64_t m_ko_hash;
};

#endif
