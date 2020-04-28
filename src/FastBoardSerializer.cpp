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

#include "FastBoardSerializer.h"
#include "FastBoard.h"

#include <cassert>
#include <cctype>
#include <boost/format.hpp>
#include <array>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "Utils.h"
#include "config.h"


std::string FastBoardSerializer::serialize_board(int lastmove) {

    int boardsize = m_board->get_boardsize();
    std::ostringstream oss;

    oss << "\n   ";
    oss << get_columns();
    for (int j = boardsize-1; j >= 0; j--) {
        oss << boost::format("%2d") % (j + 1);
        if (lastmove == m_board->get_vertex(0, j))
            oss << "(";
        else
            oss << " ";
        for (int i = 0; i < boardsize; i++) {
            if (m_board->get_state(i,j) == FastBoard::WHITE) {
                oss << "O";
            } else if (m_board->get_state(i,j) == FastBoard::BLACK)  {
                oss << "X";
            } else if (starpoint(boardsize, i, j)) {
                oss << "+";
            } else {
                oss << ".";
            }
            if (lastmove == m_board->get_vertex(i, j))
                oss << ")";
            else if (i != boardsize-1 && lastmove == m_board->get_vertex(i, j)+1)
                oss << "(";
            else oss << " ";
        }
        oss << boost::format("%2d\n") % (j + 1);
    }
    oss << "   ";
    oss << get_columns();
    oss << "\n";
    return oss.str();
}

std::string FastBoardSerializer::get_columns() {
    std::ostringstream oss;
    for (int i = 0; i < m_board->get_boardsize(); i++) {
        char c = (i < 25) ?
                (('a' + i < 'i') ? 'a' + i : 'a' + i + 1) :
                (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1);
        oss << c << " ";
    }
    oss << "\n";
    return oss.str();
}

std::string FastBoardSerializer::move_to_text(int move) const {
    std::ostringstream result;
    int size = m_board->get_boardsize();
    std::pair<int, int> coord = get_coords(move, size);

    int sidevertices = size + 2;
    int numvertices = sidevertices * sidevertices;
    if (move >= 0 && move <= numvertices) {
        int column = coord.first;
        result << static_cast<char>(column < 8 ? 'A' + column : 'A' + column + 1);
        result << (coord.second + 1);
    } else if (move == FastBoard::PASS) {
        result << "pass";
    } else if (move == FastBoard::RESIGN) {
        result << "resign";
    } else {
        result << "error";
    }

    return result.str();
}

std::string FastBoardSerializer::move_to_text_sgf(int move) const {
    std::ostringstream result;
    int size = m_board->get_boardsize();
    std::pair<int, int> coord = get_coords(move, size);

    // SGF inverts rows
    int row = size - coord.second - 1;

    int sidevertices = size + 2;
    int numvertices = sidevertices * sidevertices;
    if (move >= 0 && move <= numvertices) {
        int column = coord.first;
        if (column <= 25) {
            result << static_cast<char>('a' + column);
        } else {
            result << static_cast<char>('A' + column - 26);
        }
        if (row <= 25) {
            result << static_cast<char>('a' + row);
        } else {
            result << static_cast<char>('A' + row - 26);
        }
    } else if (move == FastBoard::PASS) {
        result << "tt";
    } else if (move == FastBoard::RESIGN) {
        result << "tt";
    } else {
        result << "error";
    }

    return result.str();
}

std::pair<int, int> FastBoardSerializer::get_coords(int move, int size) const {

    int sidevertices = size + 2;
    int column = move % sidevertices;
    int row = move / sidevertices;

    column--;
    row--;

    assert(move == FastBoard::PASS
           || move == FastBoard::RESIGN
           || (row >= 0 && row < size && column >= 0 && column <size));

    return std::make_pair(column, row);
}

int FastBoardSerializer::text_to_move(std::string move) const {
    transform(cbegin(move), cend(move), begin(move), tolower);

    if (move == "pass") {
        return FastBoard::PASS;
    } else if (move == "resign") {
        return FastBoard::RESIGN;
    } else if (move.size() < 2 || !std::isalpha(move[0]) || !std::isdigit(move[1]) || move[0] == 'i') {
        return FastBoard::NO_VERTEX;
    }

    auto column = move[0] - 'a';
    if (move[0] > 'i') {
        --column;
    }

    int row;
    std::istringstream parsestream(move.substr(1));
    parsestream >> row;
    --row;

    int size = m_board->get_boardsize();
    if (row >= size || column >= size) {
        return FastBoard::NO_VERTEX;
    }

    return m_board->get_vertex(column, row);
}

bool FastBoardSerializer::starpoint(int size, int point) {
    int stars[3];
    int points[2];
    int hits = 0;

    if (size % 2 == 0 || size < 9) {
        return false;
    }

    stars[0] = size >= 13 ? 3 : 2;
    stars[1] = size / 2;
    stars[2] = size - 1 - stars[0];

    points[0] = point / size;
    points[1] = point % size;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (points[i] == stars[j]) {
                hits++;
            }
        }
    }

    return hits >= 2;
}

bool FastBoardSerializer::starpoint(int size, int x, int y) {
    return starpoint(size, y * size + x);
}
