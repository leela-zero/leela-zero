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

#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>

#include "Timing.h"

class TimeControl {
public:
    /*
        Initialize time control. Timing info is per GTP and in centiseconds
    */
    TimeControl(int boardsize = 19,
                int maintime = 60 * 60 * 100,
                int byotime = 0, int byostones = 25,
                int byoperiods = 0);

    void start(int color);
    void stop(int color);
    int max_time_for_move(int color);
    void adjust_time(int color, int time, int stones);
    void set_boardsize(int boardsize);
    void display_times();
    void reset_clocks();

private:
    void display_color_time(int color);

    int m_maintime;
    int m_byotime;
    int m_byostones;
    int m_byoperiods;
    int m_moves_expected;

    std::array<int,  2> m_remaining_time;    /* main time per player */
    std::array<int,  2> m_stones_left;       /* stones to play in byo period */
    std::array<int,  2> m_periods_left;      /* byo periods */
    std::array<bool, 2> m_inbyo;             /* player is in byo yomi */

    std::array<Time, 2> m_times;             /* storage for player times */
};

#endif
