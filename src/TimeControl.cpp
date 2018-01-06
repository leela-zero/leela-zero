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

#include "TimeControl.h"

#include <algorithm>
#include <cassert>

#include "GTP.h"
#include "Timing.h"
#include "Utils.h"

using namespace Utils;

TimeControl::TimeControl(int boardsize, int maintime, int byotime,
                         int byostones, int byoperiods)
    : m_maintime(maintime),
      m_byotime(byotime),
      m_byostones(byostones),
      m_byoperiods(byoperiods) {

    reset_clocks();
    set_boardsize(boardsize);
}

void TimeControl::reset_clocks() {
    m_remaining_time[0] = m_maintime;
    m_remaining_time[1] = m_maintime;
    m_stones_left[0] = m_byostones;
    m_stones_left[1] = m_byostones;
    m_periods_left[0] = m_byoperiods;
    m_periods_left[1] = m_byoperiods;
    m_inbyo[0] = m_maintime <= 0;
    m_inbyo[1] = m_maintime <= 0;
    // Now that byo-yomi status is set, add time
    // back to our clocks
    if (m_inbyo[0]) {
        m_remaining_time[0] = m_byotime;
    }
    if (m_inbyo[1]) {
        m_remaining_time[1] = m_byotime;
    }
}

void TimeControl::start(int color) {
    m_times[color] = Time();
}

void TimeControl::stop(int color) {
    Time stop;
    int elapsed_centis = Time::timediff_centis(m_times[color], stop);

    assert(elapsed_centis >= 0);

    m_remaining_time[color] -= elapsed_centis;

    if (m_inbyo[color]) {
        if (m_byostones) {
            m_stones_left[color]--;
        } else if (m_byoperiods) {
            if (elapsed_centis > m_byotime) {
                m_periods_left[color]--;
            }
        }
    }

    /*
        time up, entering byo yomi
    */
    if (!m_inbyo[color] && m_remaining_time[color] <= 0) {
        m_remaining_time[color] = m_byotime;
        m_stones_left[color] = m_byostones;
        m_periods_left[color] = m_byoperiods;
        m_inbyo[color] = true;
    } else if (m_inbyo[color] && m_byostones && m_stones_left[color] <= 0) {
        // reset byoyomi time and stones
        m_remaining_time[color] = m_byotime;
        m_stones_left[color] = m_byostones;
    } else if (m_inbyo[color] && m_byoperiods) {
        m_remaining_time[color] = m_byotime;
    }
}

void TimeControl::display_color_time(int color) {
    auto rem = m_remaining_time[color] / 100;  /* centiseconds to seconds */
    auto minuteDiv = std::div(rem, 60);
    auto hourDiv = std::div(minuteDiv.quot, 60);
    auto seconds = minuteDiv.rem;
    auto minutes = hourDiv.rem;
    auto hours = hourDiv.quot;
    auto name = color == 0 ? "Black" : "White";
    myprintf("%s time: %02d:%02d:%02d", name, hours, minutes, seconds);
    if (m_inbyo[color]) {
        if (m_byostones) {
            myprintf(", %d stones left", m_stones_left[color]);
        } else if (m_byoperiods) {
            myprintf(", %d period(s) of %d seconds left",
                     m_periods_left[color], m_byotime / 100);
        }
    }
    myprintf("\n");
}

void TimeControl::display_times() {
    display_color_time(0); // Black
    display_color_time(1); // White
    myprintf("\n");
}

int TimeControl::max_time_for_move(int color) {
    // default: no byo yomi (absolute)
    auto time_remaining = m_remaining_time[color];
    auto moves_remaining = m_moves_expected;
    auto extra_time_per_move = 0;

    if (m_byotime != 0) {
        /*
          no periods or stones set means
          infinite time = 1 month
        */
        if (m_byostones == 0 && m_byoperiods == 0) {
            return 31 * 24 * 60 * 60 * 100;
        }

        // byo yomi and in byo yomi
        if (m_inbyo[color]) {
            if (m_byostones) {
                moves_remaining = m_stones_left[color];
            } else {
                assert(m_byoperiods);
                // Just use the byo yomi period
                time_remaining = 0;
                extra_time_per_move = m_byotime;
            }
        } else {
            /*
              byo yomi time but not in byo yomi yet
            */
            if (m_byostones) {
                int byo_extra = m_byotime / m_byostones;
                time_remaining = m_remaining_time[color] + byo_extra;
                // Add back the guaranteed extra seconds
                extra_time_per_move = byo_extra;
            } else {
                assert(m_byoperiods);
                int byo_extra = m_byotime * (m_periods_left[color] - 1);
                time_remaining = m_remaining_time[color] + byo_extra;
                // Add back the guaranteed extra seconds
                extra_time_per_move = m_byotime;
            }
        }
    }

    // always keep a cfg_lagbugger_cs centisecond margin
    // for network hiccups or GUI lag
    auto base_time = std::max(time_remaining - cfg_lagbuffer_cs, 0) /
                     std::max(moves_remaining, 1);
    auto inc_time = std::max(extra_time_per_move - cfg_lagbuffer_cs, 0);

    return base_time + inc_time;
}

void TimeControl::adjust_time(int color, int time, int stones) {
    m_remaining_time[color] = time;
    // From pachi: some GTP things send 0 0 at the end of main time
    if (!time && !stones) {
        m_inbyo[color] = true;
        m_remaining_time[color] = m_byotime;
        m_stones_left[color] = m_byostones;
        m_periods_left[color] = m_byoperiods;
    }
    if (stones) {
        // stones are only given in byo-yomi
        m_inbyo[color] = true;
    }
    // we must be in byo-yomi before interpreting stones
    // the previous condition guarantees we do this if != 0
    if (m_inbyo[color]) {
        if (m_byostones) {
            m_stones_left[color] = stones;
        } else if (m_byoperiods) {
            // KGS extension
            m_periods_left[color] = stones;
        }
    }
}

void TimeControl::set_boardsize(int boardsize) {
    // Note this is constant as we play, so it's fair
    // to underestimate quite a bit.
    m_moves_expected = (boardsize * boardsize) / 5;
}
