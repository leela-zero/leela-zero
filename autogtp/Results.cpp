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

#include "Results.h"
#include "Game.h"
#include "SPRT.h"
#include <QString>
#include <boost/format.hpp>
#include <iostream>

void Results::addGameResult(Sprt::GameResult result, int side) {
    m_gamesPlayed++;
    if (result == Sprt::GameResult::Win) {
        if (side == Game::BLACK)
            m_blackWins++;
        else
            m_whiteWins++;
    } else {
        if (side == Game::BLACK)
            m_blackLosses++;
        else
            m_whiteLosses++;
    }
}

std::string winPercentColumn(int wins, int games) {
    return (boost::format(" %4d %5.2f%%") % wins % (100.0 * wins / games)).str();
}

void Results::printResults(QString firstNetName, QString secondNetName) {
    /*
    Produces reports in this format.
    leelaz-ABCD1234 v leelaz-DEFG5678 (176 games)
                    wins          black       white
    leelaz-ABDC1234   65 36.93%   37 42.53%   28 31.46%
    leelaz-DEFG5678  111 63.07%   61 68.54%   50 57.47%
                                  98 55.68%   78 44.32%
    */
    auto first_name = firstNetName.left(8).toLocal8Bit().constData();
    auto second_name = secondNetName.left(8).toLocal8Bit().constData();

    // Results for player one
    auto p1_wins = m_blackWins + m_whiteWins;
    auto p1_losses = m_blackLosses + m_whiteLosses;

    auto p1_black_games = m_blackWins + m_blackLosses;
    auto p1_white_games = m_whiteWins + m_whiteLosses;

    // Results for black vs white
    auto black_wins = m_blackWins + m_whiteLosses;
    auto white_wins = m_whiteWins + m_blackLosses;

    std::cout
        << first_name << " v " << second_name
        << " ( " << m_gamesPlayed << " games)" << std::endl;
    std::cout
        << boost::format("%13s %-11s %-11s %s\n")
            % "" /* name */ % "wins" % "black" % "white";
    std::cout
        << first_name
        << winPercentColumn(p1_wins, m_gamesPlayed)
        << winPercentColumn(m_blackWins, p1_black_games)
        << winPercentColumn(m_whiteWins, p1_white_games) << std::endl;
    std::cout
        << second_name
        << winPercentColumn(p1_losses, m_gamesPlayed)
        << winPercentColumn(m_whiteLosses, p1_black_games)
        << winPercentColumn(m_blackLosses, p1_white_games) << std::endl;
    std::cout
        << std::string(20, ' ')
        << winPercentColumn(black_wins, m_gamesPlayed)
        << winPercentColumn(white_wins, m_gamesPlayed) << std::endl;
}
