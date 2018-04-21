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

#ifndef NETRESULT_H_INCLUDED
#define NETRESULT_H_INCLUDED

#include <limits>
#include <cmath>

#include "FastState.h"
class Netresult {
    uint16_t encode(float v) const {
        std::uint16_t MAX = std::numeric_limits<std::uint16_t>::max();
        v = std::round(v * MAX);
        if(v > MAX) v = static_cast<float>(MAX);
        if(v < 0) v = 0.0;
        
        return static_cast<uint16_t>(v);
    }
    float to_float(uint16_t v) const {
        std::uint16_t MAX = std::numeric_limits<std::uint16_t>::max();
        return static_cast<float>(v) / MAX;
    }

    // 19x19 board positions, 0.0 ~ 1.0 encoded as 0~65535
    std::vector<std::uint16_t> m_policy;

    // pass, 0.0 ~ 1.0 encoded as 0~65535
    std::uint16_t m_policy_pass;

    // winrate, 0.0 ~ 1.0 encoded as 0~65535
    std::uint16_t m_winrate;

public:
    Netresult() : m_policy(BOARD_SQUARES), m_policy_pass(0), m_winrate(0) {}
    float read_policy(int index) const {
        return to_float(m_policy[index]);
    }
    float read_pass() const {
        return to_float(m_policy_pass);
    } 
    float read_winrate() const {
        return to_float(m_winrate);
    }
    void write_policy(int index, float value) {
        m_policy[index] = encode(value);
    }
    void write_pass_winrate(float pass, float winrate) {
        m_policy_pass = encode(pass);
        m_winrate = encode(winrate);
    }
};

#endif
