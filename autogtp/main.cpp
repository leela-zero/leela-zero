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
#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <chrono>
#ifdef WIN32
#include <direct.h>
#endif
#include <QCommandLineParser>
#include <iostream>
#include "Game.h"
#include "Management.h"

constexpr int AUTOGTP_VERSION = 11;

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("autogtp");
    app.setApplicationVersion(QString("v%1").arg(AUTOGTP_VERSION));

    QTimer::singleShot(0, &app, SLOT(quit()));

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

    parser.addOption(gamesNumOption);
    parser.addOption(gpusOption);
    parser.addOption(keepSgfOption);
    parser.addOption(keepDebugOption);

    // Process the actual command line arguments given by the user
    parser.process(app);
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
#ifdef WIN32
    // We need to make sure these files we need are there before calling them.
    // Otherwise it will result in a crash.
    QFileInfo curl_exe("curl.exe");
    QFileInfo gzip_exe("gzip.exe");
    QFileInfo leelaz_exe("leelaz.exe");
    if (!(curl_exe.exists() && gzip_exe.exists() && leelaz_exe.exists())) {
        char cwd[_MAX_PATH];
        _getcwd(cwd, _MAX_PATH);
        cerr << "Autogtp cannot run as required executables ";
        cerr << "(curl.exe, gzip.exe and leelaz.exe) are not found in the ";
        cerr << "following folder: " << endl;
        cerr << cwd << endl;
        cerr << "Press a key to exit..." << endl;
        getchar();
        return EXIT_FAILURE;
    }
#endif
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
    QMutex mutex;
    Management boss(gpusNum, gamesNum, gpusList, AUTOGTP_VERSION,
                    parser.value(keepSgfOption), parser.value(keepDebugOption),
                    &mutex);
    boss.giveAssignments();
    mutex.lock();
    cerr.flush();
    cout.flush();
    mutex.unlock();
    return app.exec();
}
