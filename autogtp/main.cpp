/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto
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

#include <QtCore/QCoreApplication>
#include <QtCore/QTimer>
#include <QtCore/QTextStream>
#include <QtCore/QStringList>
#include <QCommandLineParser>
#include <QProcess>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QtWidgets/QShortcut>
#include <QDebug>
#include <chrono>
#ifdef WIN32
#include <direct.h>
#endif
#include <QCommandLineParser>
#include <iostream>
#include "Game.h"
#include "Management.h"
#include "Console.h"

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("autogtp");
    app.setApplicationVersion(QString("v%1").arg(AUTOGTP_VERSION));

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption gamesNumOption(
        {"g", "gamesNum"},
              "Play 'gamesNum' games on one GPU at the same time.",
              "num", "1");
    QCommandLineOption gpusOption(
        {"u", "gpus"},
              "Index of the GPU to use for multiple GPUs support.",
              "num");
    QCommandLineOption keepSgfOption(
        {"k", "keepSgf" },
              "Save SGF files after each self-play game.",
              "output directory");
    QCommandLineOption keepDebugOption(
        { "d", "debug" }, "Save training and extra debug files after each self-play game.",
                          "output directory");
    QCommandLineOption timeoutOption(
        { "t", "timeout" }, "Save running games after the timeout (in minutes) is passed and then exit.",
                          "time in minutes");

    QCommandLineOption singleOption(
        { "s", "single" }, "Exit after the first game is completed.",
                          "");

    QCommandLineOption maxOption(
        { "m", "maxgames" }, "Exit after the given number of games is completed.",
                          "max number of games");

    QCommandLineOption eraseOption(
        { "e", "erase" }, "Erase old networks when new ones are available.",
                          "");

    parser.addOption(gamesNumOption);
    parser.addOption(gpusOption);
    parser.addOption(keepSgfOption);
    parser.addOption(keepDebugOption);
    parser.addOption(timeoutOption);
    parser.addOption(singleOption);
    parser.addOption(maxOption);
    parser.addOption(eraseOption);

    // Process the actual command line arguments given by the user
    parser.process(app);
    int gamesNum = parser.value(gamesNumOption).toInt();
    QStringList gpusList = parser.values(gpusOption);
    int gpusNum = gpusList.count();
    if (gpusNum == 0) {
        gpusNum = 1;
    }
    int maxNum = -1;
    if (parser.isSet(maxOption)) {
        maxNum = parser.value(maxOption).toInt();
        if (maxNum == 0) {
            maxNum = 1;
        }
        if (maxNum < gpusNum * gamesNum) {
            gamesNum = maxNum / gpusNum;
            if (gamesNum == 0) {
                gamesNum = 1;
                gpusNum = 1;
            }
        }
        maxNum -= (gpusNum * gamesNum);
    }
    if (parser.isSet(singleOption)) {
        gamesNum = 1;
        gpusNum = 1;
        maxNum = 0;
    }

    // Map streams
    QTextStream cerr(stderr, QIODevice::WriteOnly);
    cerr << "AutoGTP v" << AUTOGTP_VERSION << endl;
    cerr << "Using " << gamesNum << " thread(s) for GPU(s)." << endl;
    if (parser.isSet(keepSgfOption)) {
        if (!QDir().mkpath(parser.value(keepSgfOption))) {
            cerr << "Couldn't create output directory for self-play SGF files!"
                 << endl;
            return EXIT_FAILURE;
        }
    }
    if (parser.isSet(keepDebugOption)) {
        if (!QDir().mkpath(parser.value(keepDebugOption))) {
            cerr << "Couldn't create output directory for self-play Debug files!"
                 << endl;
            return EXIT_FAILURE;
        }
    }
    Console *cons = nullptr;
    if (!QDir().mkpath("networks")) {
        cerr << "Couldn't create the directory for the networks files!"
             << endl;
        return EXIT_FAILURE;
    }
    Management *boss = new Management(gpusNum, gamesNum, gpusList, AUTOGTP_VERSION, maxNum,
                                      parser.isSet(eraseOption), parser.value(keepSgfOption),
                                      parser.value(keepDebugOption));
    QObject::connect(&app, &QCoreApplication::aboutToQuit, boss, &Management::storeGames);
    QTimer *timer = new QTimer();
    boss->giveAssignments();
    if (parser.isSet(timeoutOption)) {
        QObject::connect(timer, &QTimer::timeout, &app, &QCoreApplication::quit);
        timer->start(parser.value(timeoutOption).toInt() * 60000);
    } else {
        if (parser.isSet(singleOption) || parser.isSet(maxOption)) {
            QObject::connect(boss, &Management::sendQuit, &app, &QCoreApplication::quit);
        } else {
            cons = new Console();
            QObject::connect(cons, &Console::sendQuit, &app, &QCoreApplication::quit);
        }
    }
    return app.exec();
}
