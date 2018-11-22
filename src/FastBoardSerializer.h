/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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

#ifndef LEELAZ_FASTBOARDSERIALIZER_H
#define LEELAZ_FASTBOARDSERIALIZER_H

#include "config.h"

#include <array>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class FastBoard;


class FastBoardSerializer {

public:
    explicit FastBoardSerializer(FastBoard *board) {
        m_board = board;
    }

    std::string move_to_text(int move) const;
    std::string move_to_text_sgf(int move) const;
    int text_to_move(std::string move) const;

    std::string serialize_board(int lastmove = -1);

    static bool starpoint(int size, int point);
    static bool starpoint(int size, int x, int y);

private:
    FastBoard *m_board;

    std::pair<int, int> get_coords(int move, int size) const;
    std::string get_columns();
};

#endif