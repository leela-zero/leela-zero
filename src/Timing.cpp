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

#include "Timing.h"

#include <chrono>


int Time::timediff_centis(Time start, Time end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (end.m_time - start.m_time).count() / 10;
}

double Time::timediff_seconds(Time start, Time end) {
    return std::chrono::duration<double>(end.m_time - start.m_time).count();
}

Time::Time(void) {
    m_time = std::chrono::steady_clock::now();
}
