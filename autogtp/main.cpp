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
#include <QThread>
#include <chrono>
#include <iostream>
#include <cmath>
#include "Game.h"

constexpr int AUTOGTP_VERSION = 4;

// Minimal Leela Zero version we expect to see
const VersionTuple min_leelaz_version{0, 6};

constexpr int RETRY_DELAY_MIN = 30; //seconds
constexpr int RETRY_DELAY_MAX = 3600;
constexpr int MAX_RETRIES = 100;
constexpr int BATCH_UPLOAD_DELAY = 3;

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
        return false;
    }
    QString outhash = outlst[0];
    QString client_version = outlst[1];
    auto server_expected = client_version.toInt();
    if (server_expected > AUTOGTP_VERSION) {
        cerr << "Server requires client version " << server_expected
             << " but we are version " << AUTOGTP_VERSION << endl;
        cerr << "Check https://github.com/gcp/leela-zero for updates." << endl;
        exit(EXIT_FAILURE);
    }
    cerr << "Best network hash: " << outhash << endl;
    cerr << "Required client version: " << server_expected << " (OK)" << endl;
    nethash = outhash;
    return true;
}

bool network_exists(QString& netname) {
    return QFileInfo::exists(netname);
}

bool fetch_best_network(QTextStream& cerr, QString& netname) {
    if (network_exists(netname)) {
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

bool process_data(QTextStream& cerr, QString netname, QString sgf_output_path, bool upload) {
    // Uploads all stored games if 'upload' = true.
    // Otherwise deletes all of the stored games without uploading.

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
        if (!sgf_output_path.isEmpty()) {
            QString filepath = sgf_output_path + '/' + fileInfo.fileName();
            if (!QFile::exists(filepath)) {
                QFile(sgf_file).copy(filepath);
            }
        }

        // Cut .sgf, add .txt.0.gz
        data_file.chop(4);
        data_file += ".txt.0.gz";

        if (!upload) {
            cerr << "Deleting old game: " << sgf_file << endl;
            dir.remove(sgf_file);
            dir.remove(data_file);
            continue;
        }

        if (i > 0) {
            //Wait before every upload when uploading a batch
            QThread::sleep(BATCH_UPLOAD_DELAY);
        }

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

        if (outstr.contains("An error occurred")) {
            cerr << "Game upload failed." << endl;
            //Abort the upload and try again later
            break;
        }

        dir.remove(sgf_file);
        dir.remove(data_file);
    }
    return true;
}

bool upload_data(QTextStream& cerr, QString netname, QString sgf_output_path) {
    return process_data(cerr, netname, sgf_output_path, true);
}

bool delete_data(QTextStream& cerr) {
    return process_data(cerr, "", "", false);
}

bool update_network(QTextStream& cerr, QString &netname) {
    auto retries = 0;
    auto prev_netname = netname;
    while (retries++ < MAX_RETRIES) {
        if (!fetch_best_network_hash(cerr, netname)) {
            cerr << "Failed to get the best network from server." << endl;

            if (network_exists(netname)) {
                cerr << "Using the previous network." << endl;
            } else {
                auto retry_delay = std::min((unsigned)(RETRY_DELAY_MIN * std::pow(1.5, retries-1)), (unsigned)RETRY_DELAY_MAX);
                cerr << "Retrying in " <<
                    retry_delay << " s." << endl;
                QThread::sleep(retry_delay);
                continue;
            }

            return true;
        } else {
            if (prev_netname != netname) {
                //Delete any possible old gamedata to make sure we don't upload
                //old games with the new network name
                delete_data(cerr);
            }
            return fetch_best_network(cerr, netname);
        }
    }
    cerr << "Maximum number of retries exceeded. Giving up." << endl;
    exit(EXIT_FAILURE);
}

bool run_one_game(QTextStream& cerr, const QString& weightsname) {

    Game game(weightsname, cerr);
    if(!game.gameStart(min_leelaz_version)) {
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

template<typename T>
void print_timing_info(QTextStream& cerr, int games_played,
                       T start, T game_start) {
    auto game_end = std::chrono::high_resolution_clock::now();
    auto game_time_s =
        std::chrono::duration_cast<std::chrono::seconds>(game_end - game_start);
    auto total_time_s =
        std::chrono::duration_cast<std::chrono::seconds>(game_end - start);
    auto total_time_min =
        std::chrono::duration_cast<std::chrono::minutes>(total_time_s);
    cerr << games_played << " game(s) played in "
         << total_time_min.count() << " minutes = "
         << total_time_s.count() / games_played << " seconds/game"
         << ", last game took "
         << game_time_s.count() << " seconds.\n";
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    app.setApplicationName("autogtp");
    app.setApplicationVersion(QString("v%1").arg(AUTOGTP_VERSION));
    QTimer::singleShot(0, &app, SLOT(quit()));

    QCommandLineOption keep_sgf_option(
        { "k", "keep-sgf" }, "Save SGF files after each self-play game.",
                             "output directory");
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
            cerr << "Couldn't create output directory for self-play SGF files!"
                 << endl;
            return EXIT_FAILURE;
        }
    }

    auto success = true;
    auto games_played = 0;
    auto start = std::chrono::high_resolution_clock::now();
    QString netname;

    //Delete any possible stored games to avoid uploading them
    //with the new network name.
    delete_data(cerr);

    do {
        auto game_start = std::chrono::high_resolution_clock::now();
        success &= update_network(cerr, netname);
        success &= run_one_game(cerr, netname);
        success &= upload_data(cerr, netname, parser.value(keep_sgf_option));
        games_played++;
        print_timing_info(cerr, games_played, start, game_start);
    } while (success);

    cerr.flush();
    cout.flush();
    return app.exec();
}
