/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
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

#include <QtCore/QCoreApplication>
#include <QtCore/QTimer>
#include <QtCore/QTextStream>
#include <QtCore/QStringList>
#include <QCommandLineParser>
#include <QProcess>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <chrono>
#include <QCommandLineParser>
#include "../autogtp/Game.h"
#include "../autogtp/Console.h"
#include "Validation.h"

constexpr int VALIDATION_VERSION = 1;

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("validation");
    app.setApplicationVersion(QString("v%1").arg(VALIDATION_VERSION));
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption networkOption(
        {"n", "network"},
            "Networks to use as players in competition mode (two are needed).",
            "filename");
    QCommandLineOption binaryOption(
        {"b", "binary"},
            "Binary to execute for the game (default ./leelaz).",
            "filename");
    QCommandLineOption optionsOption(
        {"o", "options"},
            "Options for the binary given by -b (default '-g -p 1600 --noponder -t 1 -q -d -r 0 -w').",
            "opt_string");
    QCommandLineOption sprtOption(
        {"s", "sprt"},
            "Set the SPRT hypothesis (default '0.0:35.0').",
            "lower:upper", "0.0:35.0");
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

    parser.addOption(gamesNumOption);
    parser.addOption(gpusOption);
    parser.addOption(networkOption);
    parser.addOption(binaryOption);
    parser.addOption(optionsOption);
    parser.addOption(sprtOption);
    parser.addOption(keepSgfOption);

    // Process the actual command line arguments given by the user
    parser.process(app);
    QStringList netList = parser.values(networkOption);
    if(netList.count() != 2) {
        parser.showHelp();
    }

    QStringList binList = parser.values(binaryOption);
    while(binList.count() != 2) {
        binList << "./leelaz";
    }
    
    QStringList optsList = parser.values(optionsOption);
    while(optsList.count() != 2) {
        optsList << " -g  -p 1600 --noponder -t 1 -q -d -r 0 -w ";
    }
   
    QString sprtOpt = parser.value(sprtOption);
    QStringList sprtList = sprtOpt.split(":");
    float h0 = sprtList[0].toFloat(); 
    float h1 = sprtList[1].toFloat(); 

    int gamesNum = parser.value(gamesNumOption).toInt();
    QStringList gpusList = parser.values(gpusOption);
    int gpusNum = gpusList.count();
    if (gpusNum == 0) {
        gpusNum = 1;
    }

    QTextStream(stdout) << "validation v" << VALIDATION_VERSION << endl;
    if (parser.isSet(keepSgfOption)) {
        if (!QDir().mkpath(parser.value(keepSgfOption))) {
            QTextStream(stdout) << "Couldn't create output directory for self-play SGF files!"
                 << endl;
            return EXIT_FAILURE;
        }
    }
    QMutex mutex;
    QTextStream(stdout) << "SPRT : " << sprtOpt << " h0 " << h0 << " h1 " << h1 << endl;

    Console *cons = nullptr;
    Validation *validate = new Validation(gpusNum, gamesNum, gpusList,
                        netList.at(0), netList.at(1),
                        parser.value(keepSgfOption), &mutex,
                        binList.at(0), binList.at(1),
                        optsList.at(0), optsList.at(1),
                        h0, h1
                        );
    QObject::connect(&app, &QCoreApplication::aboutToQuit, validate, &Validation::storeSprt);
    validate->loadSprt();
    validate->startGames();
    QObject::connect(validate, &Validation::sendQuit, &app, &QCoreApplication::quit);
    cons = new Console();
    QObject::connect(cons, &Console::sendQuit, &app, &QCoreApplication::quit);
    return app.exec();
}
