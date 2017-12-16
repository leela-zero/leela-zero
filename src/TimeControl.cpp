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
    int elapsed = Time::timediff(m_times[color], stop);

    assert(elapsed >= 0);

    m_remaining_time[color] -= elapsed;

    if (m_inbyo[color]) {
        if (m_byostones) {
            m_stones_left[color]--;
        } else if (m_byoperiods) {
            if (elapsed > m_byotime) {
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

void TimeControl::display_times() {
    {
        int rem = m_remaining_time[0] / 100;  /* centiseconds to seconds */
        int hours = rem / (60 * 60);
        rem = rem % (60 * 60);
        int minutes = rem / 60;
        rem = rem % 60;
        int seconds = rem;
        myprintf("Black time: %02d:%02d:%02d", hours, minutes, seconds);
        if (m_inbyo[0]) {
            if (m_byostones) {
                myprintf(", %d stones left", m_stones_left[0]);
            } else if (m_byoperiods) {
                myprintf(", %d period(s) of %d seconds left",
                         m_periods_left[0], m_byotime / 100);
            }
        }
        myprintf("\n");
    }
    {
        int rem = m_remaining_time[1] / 100;  /* centiseconds to seconds */
        int hours = rem / (60 * 60);
        rem = rem % (60 * 60);
        int minutes = rem / 60;
        rem = rem % 60;
        int seconds = rem;
        myprintf("White time: %02d:%02d:%02d", hours, minutes, seconds);
        if (m_inbyo[1]) {
            if (m_byostones) {
                myprintf(", %d stones left", m_stones_left[1]);
            } else if (m_byoperiods) {
                myprintf(", %d period(s) of %d seconds left",
                         m_periods_left[1], m_byotime / 100);
            }
        }
        myprintf("\n");
    }
    myprintf("\n");
}

int TimeControl::max_time_for_move(int color) {
    /*
        always keep a 1 second margin for net hiccups
    */
    const int BUFFER_CENTISECS = cfg_lagbuffer_cs;

    int timealloc = 0;

    /*
        no byo yomi (absolute), easiest
    */
    if (m_byotime == 0) {
        timealloc = (m_remaining_time[color] - BUFFER_CENTISECS)
                    / m_moves_expected;
    } else if (m_byotime != 0) {
        /*
          no periods or stones set means
          infinite time = 1 month
        */
        if (m_byostones == 0 && m_byoperiods == 0) {
            return 31 * 24 * 60 * 60 * 100;
        }

        /*
          byo yomi and in byo yomi
        */
        if (m_inbyo[color]) {
            if (m_byostones) {
                timealloc = (m_remaining_time[color] - BUFFER_CENTISECS)
                             / std::max<int>(m_stones_left[color], 1);
            } else {
                assert(m_byoperiods);
                // Just use the byo yomi period
                timealloc = m_byotime - BUFFER_CENTISECS;
            }
        } else {
            /*
              byo yomi time but not in byo yomi yet
            */
            if (m_byostones) {
                int byo_extra = m_byotime / m_byostones;
                int total_time = m_remaining_time[color] + byo_extra;
                timealloc = (total_time - BUFFER_CENTISECS) / m_moves_expected;
                // Add back the guaranteed extra seconds
                timealloc += std::max<int>(byo_extra - BUFFER_CENTISECS, 0);
            } else {
                assert(m_byoperiods);
                int byo_extra = m_byotime * (m_periods_left[color] - 1);
                int total_time = m_remaining_time[color] + byo_extra;
                timealloc = (total_time - BUFFER_CENTISECS) / m_moves_expected;
                // Add back the guaranteed extra seconds
                timealloc += std::max<int>(m_byotime - BUFFER_CENTISECS, 0);
            }
        }
    }

    timealloc = std::max<int>(timealloc, 0);
    return timealloc;
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

int TimeControl::get_remaining_time(int color) {
    return m_remaining_time[color];
}
