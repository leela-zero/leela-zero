/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Seth Troisi

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

#ifndef RESULTS_H
#define RESULTS_H

#include "SPRT.h"
#include <QString>


class Results {
public:
    Results() : m_gamesPlayed(0),
                m_blackWins(0), m_blackLosses(0),
                m_whiteWins(0), m_whiteLosses(0) {}

    int getGamesPlayed() const { return m_gamesPlayed; };
    void addGameResult(Sprt::GameResult result, int side);
    void printResults(QString firstNetName, QString secondNetName);

private:
    int m_gamesPlayed;
    int m_blackWins;
    int m_blackLosses;
    int m_whiteWins;
    int m_whiteLosses;
};

#endif // RESULT_H
