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

#include <QUuid>
#include "Game.h"

Game::Game(const QString& weights, QTextStream& out) :
    QProcess(),
    output(out),
    cmdLine("./leelaz"),
    timeSettings("time_settings 0 1 0"),
    resignation(false),
    blackToMove(true),
    blackResigned(false),
    passes(0),
    moveNum(0)
{
#ifdef WIN32
    cmdLine.append(".exe");
#endif
    cmdLine.append(" -g -q -n -m 30 -r 0 -w ");
    cmdLine.append(weights);
    cmdLine.append(" -p 400 --noponder");
    fileName = QUuid::createUuid().toRfc4122().toHex();
}

bool Game::sendGtpCommand(QString cmd) {
    write(qPrintable(cmd.append("\n")));
    waitForBytesWritten(-1);
    if (!waitReady()) {
        return false;
    }
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    Q_ASSERT(readCount > 0);
    Q_ASSERT(readBuffer[0] == '=');
    // Eat double newline from GTP protocol
    if (!waitReady()) {
        return false;
    }
    readCount = readLine(readBuffer, 256);
    Q_ASSERT(readCount > 0);
    return true;
}

void Game::gameStart() {
    start(cmdLine);
    waitForStarted();
    output << "Engine has started." << endl;
    sendGtpCommand(timeSettings);
    output << "Infinite thinking time set." << endl;
}

void Game::move() {
    moveNum++;
    QString moveCmd;
    if (blackToMove) {
        moveCmd = "genmove b\n";
    } else {
        moveCmd = "genmove w\n";
    }
    write(qPrintable(moveCmd));
    waitForBytesWritten(-1);
}

bool Game::waitReady()
{
    while (!canReadLine() && state() == QProcess::Running) {
        waitForReadyRead(-1);
    }
    // somebody crashed
    if (state() != QProcess::Running) {
        return false;
    }
    return true;
}

bool Game::readMove() {
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    if (readCount <= 3 || readBuffer[0] != '=') {
        output << "Error read " << readCount << " '";
        output << readBuffer << "'" << endl;
        terminate();
        return false;
    }
    // Skip "= "
    QString moveDone = readBuffer;
    moveDone.remove(0, 2);
    moveDone = moveDone.simplified();

    // Eat double newline from GTP protocol
    if (!waitReady()) {
        return false;
    }
    readCount = readLine(readBuffer, 256);
    Q_ASSERT(readCount > 0);
    output << moveNum << " (" << moveDone << ") ";
    output.flush();
    if (moveDone.compare(QStringLiteral("pass"),
                          Qt::CaseInsensitive) == 0) {
        passes++;
    } else if (moveDone.compare(QStringLiteral("resign"),
                                 Qt::CaseInsensitive) == 0) {
        resignation = true;
        blackResigned = blackToMove;
    } else {
        passes = 0;
    }
    return true;
}

bool Game::nextMove() {
    if(resignation || passes > 1 || moveNum > (19 * 19 * 2)) {
        return false;
    }
    blackToMove = !blackToMove;
    return true;
}


bool Game::getScore() {
    if(resignation) {
        if (blackResigned) {
            winner = QString(QStringLiteral("white"));
        } else {
            winner = QString(QStringLiteral("black"));
        }
    } else{
        write("final_score\n");
        waitForBytesWritten(-1);
        if (!waitReady()) {
            return false;
        }
        char readBuffer[256];
        readLine(readBuffer, 256);
        QString score = readBuffer;
        score.remove(0, 2);
        if (readBuffer[2] == 'W') {
            winner = QString(QStringLiteral("white"));
        } else if (readBuffer[2] == 'B') {
            winner = QString(QStringLiteral("black"));
        }
        if(!waitReady()) {
            return false;
        }
        output << "Score: " << score;
    }
    if (winner.isNull()) {
        output << "No winner found" << endl;
        return false;
    }
    output << "Winner: " << winner << endl;
    return true;
}

bool Game::writeSgf() {
    output << "Writing " << fileName + ".sgf" << endl;

    if (!sendGtpCommand(qPrintable("printsgf " + fileName + ".sgf\n"))) {
        return false;
    }
    return true;
}

bool Game::dumpTraining() {
    output << "Dumping " << fileName + ".txt" << endl;

    if (!sendGtpCommand(qPrintable("dump_training " + winner +
                        " " + fileName + ".txt\n"))) {
        return false;
    }
    return true;
}

void Game::gameQuit() {
    write(qPrintable("quit\n"));
    waitForFinished(-1);
}
