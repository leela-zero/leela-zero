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
#include <QTextStream>

class Game : QProcess {
public:
    Game(QString weights, QTextStream *out);
    ~Game(){}
    void gameStart();
    void move();
    bool waitForMove() {return waitReady();}
    bool readMove();
    bool nextMove();
    bool getScore();
    bool writeSgf();
    bool dumpTraining();
    void gameQuit();

private:
    QTextStream *output;
    QString cmdLine;
    QString timeSettings;
    QString moveDone;
    QString winner;
    QString score;
    QString sgfName;
    QString trainingName;
    int readCount;
    char readBuffer[256];
    bool resignation;
    bool blackToMove;
    bool blackResigned;
    int passes;
    int moveNum;
    bool sendGtpCommand(QString cmd);
    bool waitReady();
};





#endif /* GAME_H */

