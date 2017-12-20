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

#ifndef TIMING_H_INCLUDED
#define TIMING_H_INCLUDED

#include <chrono>

class Time {
public:
    /* sets to current time */
    Time(void);

    /* time difference in centiseconds */
    static int timediff_centis(Time start, Time end);

    /* time difference in seconds */
    static double timediff_seconds(Time start, Time end);

private:
    std::chrono::steady_clock::time_point m_time;
};

#endif
