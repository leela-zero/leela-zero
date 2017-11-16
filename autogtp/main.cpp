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
#include <iostream>
#include "Game.h"

constexpr int AUTOGTP_VERSION = 2;

bool fetch_best_network_hash(QTextStream& cerr, QString& nethash) {
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    prog_cmdline.append(" http://zero.sjeng.org/best-network-hash");
    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);
    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QStringList outlst = outstr.split("\n");
    if (outlst.size() != 2) {
        cerr << "Unexpected output from server: " << endl << output << endl;
        exit(EXIT_FAILURE);
    }
    QString outhash = outlst[0];
    cerr << "Best network hash: " << outhash << endl;
    QString client_version = outlst[1];
    auto server_expected = client_version.toInt();
    cerr << "Required client version: " << client_version;
    if (server_expected > AUTOGTP_VERSION) {
        cerr << endl;
        cerr << "Server requires client version " << server_expected
             << " but we are version " << AUTOGTP_VERSION << endl;
        cerr << "Check https://github.com/gcp/leela-zero for updates." << endl;
        exit(EXIT_FAILURE);
    } else {
        cerr << " (OK)" << endl;
    }
    nethash = outhash;
    return true;
}

bool fetch_best_network(QTextStream& cerr, QString& netname) {
    if (QFileInfo::exists(netname)) {
        cerr << "Already downloaded network." << endl;
        return true;
    }

    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    // Be quiet, but output the real file name we saved to
    // Use the filename from the server
    // Resume download if file exists (aka avoid redownloading, and don't
    // error out if it exists)
    prog_cmdline.append(" -s -O -J");
    prog_cmdline.append(" -w %{filename_effective}");
    prog_cmdline.append(" http://zero.sjeng.org/best-network");

    cerr << prog_cmdline << endl;

    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QStringList outlst = outstr.split("\n");
    QString outfile = outlst[0];
    cerr << "Curl filename: " << outfile << endl;
#ifdef WIN32
    QProcess::execute("gzip.exe -d -k -q " + outfile);
#else
    QProcess::execute("gunzip -k -q " + outfile);
#endif
    // Remove extension (.gz)
    outfile.chop(3);
    cerr << "Net filename: " << outfile << endl;
    netname = outfile;

    return true;
}

bool upload_data(QTextStream& cerr, const QString& netname, QString sgf_output_path) {
    // Find output SGF and txt files
    QDir dir;
    QStringList filters;
    filters << "*.sgf";
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);

    QFileInfoList list = dir.entryInfoList();
    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fileInfo = list.at(i);
        QString sgf_file = fileInfo.fileName();
        QString data_file = sgf_file;
        // Save first if requested
        if (!sgf_output_path.isEmpty())
            QFile(sgf_file).copy(sgf_output_path + '/' + fileInfo.fileName());
        // Cut .sgf, add .txt.0.gz
        data_file.chop(4);
        data_file += ".txt.0.gz";
        // Gzip up the sgf too
#ifdef WIN32
        QProcess::execute("gzip.exe " + sgf_file);
#else
        QProcess::execute("gzip " + sgf_file);
#endif
        sgf_file += ".gz";
        QString prog_cmdline("curl");
#ifdef WIN32
        prog_cmdline.append(".exe");
#endif
        prog_cmdline.append(" -F networkhash=" + netname);
        prog_cmdline.append(" -F clientversion=" + QString::number(AUTOGTP_VERSION));
        prog_cmdline.append(" -F sgf=@" + sgf_file);
        prog_cmdline.append(" -F trainingdata=@" + data_file);
        prog_cmdline.append(" http://zero.sjeng.org/submit");
        cerr << prog_cmdline << endl;
        QProcess curl;
        curl.start(prog_cmdline);
        curl.waitForFinished(-1);
        QByteArray output = curl.readAllStandardOutput();
        QString outstr(output);
        cerr << outstr;
        dir.remove(sgf_file);
        dir.remove(data_file);
    }
    return true;
}

bool run_one_game(QTextStream& cerr, const QString& weightsname) {

    Game game(weightsname, cerr);
    if(!game.gameStart()) {
        return false;
    }
    do {
        game.move();
        if(!game.waitForMove()) {
            return false;
        }
        game.readMove();
    } while (game.nextMove());
    cerr << "Game has ended." << endl;
    if (game.getScore()) {
        game.writeSgf();
        game.dumpTraining();
    }
    cerr << "Stopping engine." << endl;
    game.gameQuit();
    return true;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    app.setApplicationName("autogtp");
    app.setApplicationVersion(QString("v%1").arg(AUTOGTP_VERSION));
    QTimer::singleShot(0, &app, SLOT(quit()));

    QCommandLineOption keep_sgf_option({ "k", "keep-sgf" }, "Save .sgf files after each self-play game.", "output directory");
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOption(keep_sgf_option);
    parser.process(app);

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

    if (parser.isSet(keep_sgf_option)) {
        if (!QDir().mkpath(parser.value(keep_sgf_option))) {
            cerr << "Couldn't created output directory for self-play .sgf files!" << endl;
            return EXIT_FAILURE;
        }
    }

    auto success = true;
    auto games_played = 0;

    do {
        QString netname;
        success &= fetch_best_network_hash(cerr, netname);
        success &= fetch_best_network(cerr, netname);
        success &= run_one_game(cerr, netname);
        success &= upload_data(cerr, netname, parser.value(keep_sgf_option));
        games_played++;
        cerr << games_played << " games played." << endl;
    } while (success);

    cerr.flush();
    cout.flush();
    return app.exec();
}
