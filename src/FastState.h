/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef FASTSTATE_H_INCLUDED
#define FASTSTATE_H_INCLUDED

#include <cstddef>
#include <array>
#include <string>
#include <vector>

#include "FullBoard.h"

class FastState {
public:
    void init_game(int size, float komi);
    void reset_game();
    void reset_board();

    void play_move(int vertex);
    bool is_move_legal(int color, int vertex) const;

    void set_komi(float komi);
    float get_komi() const;
    void set_handicap(int hcap);
    int get_handicap() const;
    int get_passes() const;
    int get_to_move() const;
    void set_to_move(int tomove);
    void set_passes(int val);
    void increment_passes();

    float final_score() const;
    std::uint64_t get_symmetry_hash(int symmetry) const;

    size_t get_movenum() const;
    int get_last_move() const;
    void display_state();
    std::string move_to_text(int move);

    FullBoard board;

    float m_komi;
    int m_handicap;
    int m_passes;
    int m_komove;
    size_t m_movenum;
    int m_lastmove;

protected:
    void play_move(int color, int vertex);
};

#endif
