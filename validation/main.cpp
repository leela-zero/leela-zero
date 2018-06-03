/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
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
    parser.addOption(sprtOption);
    parser.addOption(keepSgfOption);
    parser.addPositionalArgument(
                "engine_options",
                "Engine options specified as: [name]=\"[value]\"\n"
                "Engine options before a binary is specified are for both binaries. "
                "Engine options after a binary are for that binary only.\n"
                "names:\n"
                "binary\tBinary to execute for the game\n(default ./leelaz).\n"
                "options\tOptions for the binary\n(default \"-g -p 1600 --noponder -t 1 -q -d -r 0 -w\").\n"
                "network\tNetwork for the binary to use.\n"
                "gtp-command\tGTP command to send to the binary\n(default \"time_settings 0 1 0\").\n"
                "Multiple gtp-command engine options can be specified in the order to be sent.\n");

    parser.setOptionsAfterPositionalArgumentsMode(QCommandLineParser::ParseAsPositionalArguments);

    // Process the actual command line arguments given by the user
    parser.process(app);

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

    auto default_engine = engine_t(
        {"./leelaz", " -g  -p 1600 --noponder -t 1 -q -d -r 0 -w ", "", {"time_settings 0 1 0"}});
    auto engines = QVector<engine_t>({default_engine, default_engine});

    auto engine_idx = -1;
    const auto args = parser.positionalArguments();
    for (auto arg : args) {
        auto name = arg.section('=', 0, 0);
        auto val = arg.section('=', 1);
        if (name.isEmpty()) {
            continue;
        }

        if (name == "binary") {
            engine_idx++;
            if (engine_idx > 1) {
                parser.showHelp();
            }
            engines[engine_idx].binary = val;
        } else if (name == "options") {
            if (engine_idx == -1) {
                engines[0].options = val;
                engines[1].options = val;
            } else {
                engines[engine_idx].options = val;
            }
        } else if (name == "network") {
            if (engine_idx == -1) {
                engines[0].network = val;
                engines[1].network = val;
            } else {
                engines[engine_idx].network = val;
            }
        } else if (name == "gtp-command") {
            if (engine_idx == -1) {
                engines[0].commands.append(val);
                engines[1].commands.append(val);
            } else {
                engines[engine_idx].commands.append(val);
            }
        }
    }

    if (engines[0].network == "" || engines[1].network == "") {
        parser.showHelp();
    }

    QMutex mutex;
    QTextStream(stdout) << "SPRT : " << sprtOpt << " h0 " << h0 << " h1 " << h1 << endl;

    Console *cons = nullptr;
    Validation *validate = new Validation(gpusNum, gamesNum, gpusList,
                                          engines,
                                          parser.value(keepSgfOption), &mutex,
                                          h0, h1);
    QObject::connect(&app, &QCoreApplication::aboutToQuit, validate, &Validation::storeSprt);
    validate->loadSprt();
    validate->startGames();
    QObject::connect(validate, &Validation::sendQuit, &app, &QCoreApplication::quit);
    cons = new Console();
    QObject::connect(cons, &Console::sendQuit, &app, &QCoreApplication::quit);
    return app.exec();
}
