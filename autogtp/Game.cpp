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

Game::Game(const QString& weights, const QString& opt) :
    QProcess(),
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
    cmdLine.append(opt);
    cmdLine.append(weights);
    cmdLine.append(" -p 1000 --noponder");
    fileName = QUuid::createUuid().toRfc4122().toHex();
}

void Game::error(int errnum) {
    output << "*ERROR*: ";
    switch(errnum) {
        case Game::NO_LEELAZ:
            QTextStream(stdout) << "No 'leelaz' binary found." << endl;
            break;
        case Game::PROCESS_DIED:
            QTextStream(stdout) << "The 'leelaz' process died unexpected." << endl;
            break;
        case Game::WRONG_GTP:
            QTextStream(stdout) << "Error in GTP response." << endl;
            break;
        case Game::LAUNCH_FAILURE:
            output << "Could not talk to engine after launching." << endl;
            break;
        default:
            QTextStream(stdout) << "Unexpected error." << endl;
            break;
    }
}

bool Game::eatNewLine() {
    char readBuffer[256];
    // Eat double newline from GTP protocol
    if (!waitReady()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    auto readCount = readLine(readBuffer, 256);
    if(readCount < 0) {
        error(Game::WRONG_GTP);
        return false;
    }
    return true;
}

bool Game::sendGtpCommand(QString cmd) {
    write(qPrintable(cmd.append("\n")));
    waitForBytesWritten(-1);
    if (!waitReady()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    Q_ASSERT(readCount > 0);
    Q_ASSERT(readBuffer[0] == '=');
    if (!eatNewLine()) {
        error(Game::PROCESS_DIED);
        return false;
    }

    Q_ASSERT(readCount > 0);
    return true;
}

void Game::checkVersion(const VersionTuple &min_version) {
    write(qPrintable("version\n"));
    waitForBytesWritten(-1);
    if (!waitReady()) {
        error(Game::LAUNCH_FAILURE);
        exit(EXIT_FAILURE);
    }
    char readBuffer[256];
    int readCount = readLine(readBuffer, 256);
    // We expect to read at last "=, space, something"
    if (readCount <= 3 || readBuffer[0] != '=') {
        output << "GTP: " << readBuffer << endl;
        error(Game::WRONG_GTP);
        exit(EXIT_FAILURE);
    }
    QString version_buff(&readBuffer[2]);
    version_buff = version_buff.simplified();
    QStringList version_list = version_buff.split(".");
    if (version_list.size() < 2) {
        output << "Unexpected Leela Zero version: " << version_buff << endl;
        exit(EXIT_FAILURE);
    }
    if (version_list[0].toInt() < std::get<0>(min_version)
        || (version_list[0].toInt() == std::get<0>(min_version)
           && version_list[1].toInt() < std::get<1>(min_version))) {
        output << "Leela version is too old, saw " << version_buff
               << " but expected "
               << std::get<0>(min_version) << "."
               << std::get<1>(min_version) << "." << endl;
        output << "Check https://github.com/gcp/leela-zero for updates."
                << endl;
        exit(EXIT_FAILURE);
    }
    if (!eatNewLine()) {
        error(Game::WRONG_GTP);
        exit(EXIT_FAILURE);
    }
}

bool Game::gameStart(const VersionTuple &min_version) {
    start(cmdLine);
    if(!waitForStarted()) {
        error(Game::NO_LEELAZ);
        return false;
    }
    // This either succeeds or we exit immediately, so no need to
    // check any return values.
    checkVersion(min_version);
    QTextStream(stdout) << "Engine has started." << endl;
    sendGtpCommand(timeSettings);
    QTextStream(stdout) << "Infinite thinking time set." << endl;
    return true;
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

bool Game::waitReady() {
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
        error(Game::WRONG_GTP);
        QTextStream(stdout) << "Error read " << readCount << " '";
        QTextStream(stdout) << readBuffer << "'" << endl;
        terminate();
        return false;
    }
    // Skip "= "
    moveDone = readBuffer;
    moveDone.remove(0, 2);
    moveDone = moveDone.simplified();
    if (!eatNewLine()) {
        error(Game::PROCESS_DIED);
        return false;
    }
    if(readCount == 0) {
        error(Game::WRONG_GTP);
    }
    QTextStream(stdout) << moveNum << " (" << moveDone << ") ";
    QTextStream(stdout).flush();
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
    blackToMove = !blackToMove;
    return true;
}

bool Game::setMove(const QString& m) {
    if (!sendGtpCommand(m)) {
        return false;
    }
    QStringList moves = m.split(" "); 
    if (moves.at(2).compare(QStringLiteral("pass"), Qt::CaseInsensitive) == 0) {
        passes++;
    } else if (moves.at(2).compare(QStringLiteral("resign"), Qt::CaseInsensitive) == 0) {
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
            error(Game::PROCESS_DIED);
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
        if (!eatNewLine()) {
            error(Game::PROCESS_DIED);
            return false;
        }
        QTextStream(stdout) << "Score: " << score;
    }
    if (winner.isNull()) {
        QTextStream(stdout) << "No winner found" << endl;
        return false;
    }
    QTextStream(stdout) << "Winner: " << winner << endl;
    return true;
}

int Game::getWinner() {
    if(winner.compare(QStringLiteral("white"), Qt::CaseInsensitive) == 0)
        return Game::WHITE;
    else
        return Game::BLACK;
}
bool Game::writeSgf() {
    QTextStream(stdout) << "Writing " << fileName + ".sgf" << endl;

    if (!sendGtpCommand(qPrintable("printsgf " + fileName + ".sgf"))) {
        return false;
    }
    return true;
}

bool Game::dumpTraining() {
    QTextStream(stdout) << "Dumping " << fileName + ".txt" << endl;

    if (!sendGtpCommand(qPrintable("dump_training " + winner +
                        " " + fileName + ".txt"))) {
        return false;
    }
    return true;
}

void Game::gameQuit() {
    write(qPrintable("quit\n"));
    waitForFinished(-1);
}
