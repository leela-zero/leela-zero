/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Marco Calignano

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

#ifndef GAME_H
#define GAME_H

#include <QProcess>
#include <tuple>

using VersionTuple = std::tuple<int, int>;

class Game : QProcess {
public:
    Game(const QString& weights,
         const QString& opt);
    ~Game() = default;
    bool gameStart(const VersionTuple& min_version);
    void move();
    bool waitForMove() { return waitReady(); }
    bool readMove();
    bool nextMove();
    bool getScore();
    bool writeSgf();
    bool fixSgf(QString& weightFile, bool resignation);
    bool dumpTraining();
    bool dumpDebug();
    void gameQuit();
    QString getMove() const { return m_moveDone; }
    QString getFile() const { return m_fileName; }
    bool setMove(const QString& m);
    bool checkGameEnd();
    void setCmdLine(const QString& cmd)  { m_cmdLine = cmd; }
    int getWinner();
    QString getWinnerName() const { return m_winner; }
    int getMovesCount() const { return m_moveNum; }
    QString getResult() const { return m_result.trimmed(); }
    enum {
        BLACK = 0,
        WHITE = 1,
    };

private:
    enum {
        NO_LEELAZ = 1,
        PROCESS_DIED,
        WRONG_GTP,
        LAUNCH_FAILURE
    };
    QString m_cmdLine;
    QString m_timeSettings;
    QString m_winner;
    QString m_fileName;
    QString m_moveDone;
    QString m_result;
    bool m_resignation;
    bool m_blackToMove;
    bool m_blackResigned;
    int m_passes;
    int m_moveNum;
    bool sendGtpCommand(QString cmd);
    void checkVersion(const VersionTuple &min_version);
    bool waitReady();
    bool eatNewLine();
    void error(int errnum);
};

#endif /* GAME_H */
