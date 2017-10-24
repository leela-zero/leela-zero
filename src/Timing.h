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

#include "config.h"

#include <time.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

class Time {
public:
    /*
        sets to current time
    */
    Time(void);

    /*
        time difference in centiseconds
    */
    static int timediff(Time start, Time end);

private:
    rtime_t m_time;
};

#endif
