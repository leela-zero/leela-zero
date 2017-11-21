/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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
#include <iostream>
#include "Game.h"
#include "SPRT.h"
#include "Validation.h"
#include "Production.h"

constexpr int AUTOGTP_VERSION = 4;
// Minimal Leela Zero version we expect to see
const VersionTuple min_leelaz_version{0, 6};

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("autogtp");
    app.setApplicationVersion(QString("v%1").arg(AUTOGTP_VERSION));

    QTimer::singleShot(0, &app, SLOT(quit()));

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption competitionOption(
        {"c", "competition"}, "Play two networks against each other.");
    QCommandLineOption networkOption(
        {"n", "network"},
            "Networks to use as players in competition mode (two are needed).",
            "filename");
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

    parser.addOption(competitionOption);
    parser.addOption(gamesNumOption);
    parser.addOption(gpusOption);
    parser.addOption(networkOption);
    parser.addOption(keepSgfOption);

    // Process the actual command line arguments given by the user
    parser.process(app);
    bool competition  = parser.isSet(competitionOption);
    QStringList netList = parser.values(networkOption);
    if(competition && netList.count() != 2) {
        parser.showHelp();
    }
    int gamesNum = parser.value(gamesNumOption).toInt();
    QStringList gpusList = parser.values(gpusOption);
    int gpusNum = gpusList.count();
    if (gpusNum == 0) {
        gpusNum = 1;
    }

    // Map streams
    QTextStream cin(stdin, QIODevice::ReadOnly);
    QTextStream cout(stdout, QIODevice::WriteOnly);
#if defined(LOG_ERRORS_TO_FILE)
    // Log stderr to file
    QFile caFile("output.txt");
    caFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append);
    if(!caFile.isOpen()){
        qDebug() << "- Error, unable to open" << "outputFilename" << "for output";
    }
    QTextStream cerr(&caFile);
#else
    QTextStream cerr(stderr, QIODevice::WriteOnly);
#endif
    cerr << "autogtp v" << AUTOGTP_VERSION << endl;
    if (parser.isSet(keepSgfOption)) {
        if (!QDir().mkpath(parser.value(keepSgfOption))) {
            cerr << "Couldn't create output directory for self-play SGF files!"
                 << endl;
            return EXIT_FAILURE;
        }
    }
    QMutex mutex;
    if(competition) {
        Validation validate(gpusNum, gamesNum, gpusList,
                            netList.at(0), netList.at(1), &mutex);
        validate.startGames();
        mutex.lock();
    } else {
        Production prod(gpusNum, gamesNum, gpusList, AUTOGTP_VERSION,
                        parser.value(keepSgfOption), &mutex);
        prod.startGames();
        mutex.lock();
    }
    cerr.flush();
    cout.flush();
    return app.exec();
}
