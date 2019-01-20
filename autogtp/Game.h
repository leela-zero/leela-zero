/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Marco Calignano

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

#include <QFileInfo>
#include <QProcess>
#include <tuple>

using VersionTuple = std::tuple<int, int, int>;

class Engine {
public:
    Engine(const QString& network,
           const QString& options,
           const QStringList& commands = QStringList("time_settings 0 1 0"),
           const QString& binary = QString("./leelaz")) :
        m_binary(binary), m_options(options),
        m_network(network), m_commands(commands) {
#ifdef WIN32
        m_binary.append(".exe");
#endif
        if (!QFileInfo::exists(m_binary)) {
            m_binary.remove(0, 2); // ./leelaz -> leelaz
        }
    }
    Engine() = default;
    QString getCmdLine(void) const {
        return m_binary + " " + m_options + " " + m_network;
    }
    QString getNetworkFile(void) const {
        return QFileInfo(m_network).baseName();
    }
    QString m_binary;
    QString m_options;
    QString m_network;
    QStringList m_commands;
};

class Game : QProcess {
public:
    Game(const Engine& engine);
    ~Game() = default;
    bool gameStart(const VersionTuple& min_version,
                   const QString &sgf = QString(),
                   const int moves = 0);
    void move();
    bool waitForMove() { return waitReady(); }
    bool readMove();
    bool nextMove();
    bool getScore();
    bool loadSgf(const QString &fileName);
    bool loadSgf(const QString &fileName, const int moves);
    bool writeSgf();
    bool loadTraining(const QString &fileName);
    bool saveTraining();
    bool fixSgf(const Engine& whiteEngine, const bool resignation,
        const bool isSelfPlay);
    bool dumpTraining();
    bool dumpDebug();
    void gameQuit();
    QString getMove() const { return m_moveDone; }
    QString getFile() const { return m_fileName; }
    bool setMove(const QString& m);
    bool checkGameEnd();
    int getWinner();
    QString getWinnerName() const { return m_winner; }
    int getMovesCount() const { return m_moveNum; }
    void setMovesCount(int moves);
    int getToMove() const { return m_blackToMove ? BLACK : WHITE; }
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
    Engine m_engine;
    QString m_winner;
    QString m_fileName;
    QString m_moveDone;
    QString m_result;
    bool m_isHandicap;
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
    void fixSgfPlayer(QString& sgfData, const Engine& whiteEngine);
    void fixSgfComment(QString& sgfData, const Engine& whiteEngine,
        const bool isSelfPlay);
    void fixSgfResult(QString& sgfData, const bool resignation);
};

#endif /* GAME_H */
