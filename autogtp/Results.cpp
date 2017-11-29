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
#include <QTextStream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

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

void Results::printResults(QString firstNetName, QString secondNetName) {
        /*
        Produces reports in this format.
        leelaz-ABCD1234 v leelaz-DEFG5678 (176 games)
                        wins          black       white
        leelaz-ABDC1234   65 36.93%   37 42.53%   28 31.46%
        leelaz-DEFG5678  111 63.07%   61 68.54%   50 57.47%
                                      98 55.68%   78 44.32%
        */

        QString first_name = firstNetName.left(8);
        QString second_name = secondNetName.left(8);

        // Results for player one
        auto p1_wins = m_blackWins + m_whiteWins;
        auto p1_losses = m_blackLosses + m_whiteLosses;

        auto p1_black_games = m_blackWins + m_blackLosses;
        auto p1_white_games = m_whiteWins + m_whiteLosses;

        // Results for black vs white
        auto black_wins = m_blackWins + m_whiteLosses;
        auto white_wins = m_whiteWins + m_blackLosses;

        QTextStream(stdout) << QString("\n%1 v %2 (%3 games)\n")
             .arg(first_name, second_name).arg(m_gamesPlayed);

        QTextStream(stdout) <<
            str(boost::format("%-14s %-14s %-14s %s\n")
                % "" /* name */ % "wins" % "black" % "white").c_str();
        QTextStream(stdout) <<
            str(boost::format("%-9s %4d %5.2f%% %4d %5.2f%% %4d %5.2f%%\n")
                % first_name.toLocal8Bit().constData()
                % p1_wins
                % (100.0 * p1_wins / m_gamesPlayed)
                % m_blackWins
                % (100.0 * m_blackWins / p1_black_games)
                % m_whiteWins
                % (100.0 * m_whiteWins / p1_white_games)).c_str();
        QTextStream(stdout) <<
            str(boost::format( "%-9s %4d %5.2f%% %4d %5.2f%% %4d %5.2f%%\n")
                % second_name.toLocal8Bit().constData()
                % p1_losses
                % (100.0 * p1_losses / m_gamesPlayed)
                % m_whiteLosses
                % (100.0 * m_whiteLosses / p1_white_games)
                % m_blackLosses
                % (100.0 * m_blackLosses / p1_white_games)).c_str();
        QTextStream(stdout) <<
            str(boost::format("%-9s %11s %4d %5.2f%% %4d %5.2f%%\n")
                % "" /* name */
                % "" /* wins column */
                % black_wins
                % (100.0 * black_wins / m_gamesPlayed)
                % white_wins
                % (100.0 * white_wins / m_gamesPlayed)).c_str();
}
