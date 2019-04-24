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

#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>
#include <memory>

#include "config.h"
#include "Timing.h"

class TimeControl {
public:
    /*
        Initialize time control. Timing info is per GTP and in centiseconds
    */
    TimeControl(int maintime = 60 * 60 * 100,
                int byotime = 0, int byostones = 0,
                int byoperiods = 0);

    void start(int color);
    void stop(int color);
    int max_time_for_move(int boardsize, int color, size_t movenum) const;
    void adjust_time(int color, int time, int stones);
    void display_times();
    void reset_clocks();
    bool can_accumulate_time(int color) const;
    size_t opening_moves(int boardsize) const;
    std::string to_text_sgf() const;
    static std::shared_ptr<TimeControl> make_from_text_sgf(
        const std::string& maintime, const std::string& byoyomi,
        const std::string& black_time_left, const std::string& white_time_left,
        const std::string& black_moves_left, const std::string& white_moves_left);
private:
    std::string stones_left_to_text_sgf(const int color) const;
    void display_color_time(int color);
    int get_moves_expected(int boardsize, size_t movenum) const;

    int m_maintime;
    int m_byotime;
    int m_byostones;
    int m_byoperiods;

    std::array<int,  2> m_remaining_time;    /* main time per player */
    std::array<int,  2> m_stones_left;       /* stones to play in byo period */
    std::array<int,  2> m_periods_left;      /* byo periods */
    std::array<bool, 2> m_inbyo;             /* player is in byo yomi */

    std::array<Time, 2> m_times;             /* storage for player times */
};

#endif
