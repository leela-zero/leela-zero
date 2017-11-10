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
#include <QtCore/QPair>
#include <QtCore/QVector>
#include <QProcess>
#include <QFile>
#include <QDir>
#include <QRegularExpression>
#include <QUuid>
#include <QDebug>
#include <iostream>
#include <functional>

bool waitForReadyRead(QProcess& process) {
    while (!process.canReadLine() && process.state() == QProcess::Running) {
        process.waitForReadyRead(-1);
    }

    // somebody crashed
    if (process.state() != QProcess::Running) {
        return false;
    }

    return true;
}

bool sendGtpCommand(QProcess& proc, QString cmd) {
    QString cmdEndl(cmd);
    cmdEndl.append(qPrintable("\n"));

    proc.write(qPrintable(cmdEndl));
    proc.waitForBytesWritten(-1);
    if (!waitForReadyRead(proc)) {
        return false;
    }
    char readbuff[256];
    auto read_cnt = proc.readLine(readbuff, 256);
    Q_ASSERT(read_cnt > 0);
    Q_ASSERT(readbuff[0] == '=');
    // Eat double newline from GTP protocol
    if (!waitForReadyRead(proc)) {
        return false;
    }
    read_cnt = proc.readLine(readbuff, 256);
    Q_ASSERT(read_cnt > 0);
    return true;
}

bool fetch_best_network_hash(QTextStream& cerr, QString& netname) {
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    prog_cmdline.append(" http://zero-test.sjeng.org/best-network-hash");
        QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);
    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QStringList outlst = outstr.split("\n");
    QString outhash = outlst[0];
    cerr << "Best network hash: " << outhash << endl;
    netname = outhash;
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
    prog_cmdline.append(" -s -C - -O -J");
    prog_cmdline.append(" -w %{filename_effective}");
    prog_cmdline.append(" http://zero-test.sjeng.org/best-network");

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
    QProcess::execute("gunzip.exe -k -q " + outfile);
#else
    QProcess::execute("gunzip -k -q " + outfile);
#endif
    // Remove extension (.gz)
    outfile.chop(3);
    cerr << "Net filename: " << outfile << endl;
    netname = outfile;

    return true;
}

bool upload_data(QTextStream& cerr, QString& netname) {
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
        prog_cmdline.append(" -F sgf=@" + sgf_file);
        prog_cmdline.append(" -F trainingdata=@" + data_file);
        prog_cmdline.append(" http://zero-test.sjeng.org/submit");
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

bool run_one_game(QTextStream& cerr, QString weightsname) {
    QString prog_cmdline("./leelaz");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    prog_cmdline.append(" -g -q -n -m 30 -r 0 -w ");
    prog_cmdline.append(weightsname);
    prog_cmdline.append(" -p 400 --noponder");

    cerr << prog_cmdline << endl;

    QProcess first_process, second_process;
    first_process.start(prog_cmdline);
    second_process.start(prog_cmdline);

    first_process.waitForStarted();
    second_process.waitForStarted();

    char readbuff[256];
    int read_cnt;

    QString winner;
    bool stop = false;
    bool black_to_move = true;
    bool black_resigned = false;
    bool first_to_move = true;
    int passes = 0;
    int move_num = 0;

    // Set infinite time
    if (!sendGtpCommand(first_process,
                        QStringLiteral("time_settings 0 1 0"))) {
        return false;
    }
    if (!sendGtpCommand(second_process,
                        QStringLiteral("time_settings 0 1 0"))) {
        return false;
    }
    cerr << "Infinite thinking time set." << endl;

    do {
        move_num++;
        QString move_cmd;
        if (black_to_move) {
            move_cmd = "genmove b\n";
        } else {
            move_cmd = "genmove w\n";
        }
        /// Send genmove to the right process
        auto proc = std::ref(first_process);
        if (!first_to_move) {
            proc = std::ref(second_process);
        }
        proc.get().write(qPrintable(move_cmd));
        proc.get().waitForBytesWritten(-1);
        if (!waitForReadyRead(proc)) {
            return false;
        }
        // Eat response
        read_cnt = proc.get().readLine(readbuff, 256);
        if (read_cnt <= 3 || readbuff[0] != '=') {
            cerr << "Error read " << read_cnt
                 << " '" << readbuff << "'" << endl;
            second_process.terminate();
            first_process.terminate();
            return false;
        }
        // Skip "= "
        QString resp_move(&readbuff[2]);
        resp_move = resp_move.simplified();

        // Eat double newline from GTP protocol
        if (!waitForReadyRead(proc)) {
            return false;
        }
        read_cnt = proc.get().readLine(readbuff, 256);
        Q_ASSERT(read_cnt > 0);

        cerr << move_num << " (" << resp_move << ") ";
        cerr.flush();

        QString move_side(QStringLiteral("play "));
        QString side_prefix;

        if (black_to_move) {
            side_prefix = QStringLiteral("b ");
        } else {
            side_prefix = QStringLiteral("w ");
        }

        move_side += side_prefix + resp_move + "\n";

        if (resp_move.compare(QStringLiteral("pass"),
                              Qt::CaseInsensitive) == 0) {
            passes++;
        } else if (resp_move.compare(QStringLiteral("resign"),
                                     Qt::CaseInsensitive) == 0) {
            passes++;
            stop = true;
            black_resigned = black_to_move;
        } else {
            passes = 0;
        }

        // Got move, swap sides now
        first_to_move = !first_to_move;
        black_to_move = !black_to_move;

        if (!stop) {
            auto next = std::ref(first_process);
            if (!first_to_move) {
                next = std::ref(second_process);
            }
            if (!sendGtpCommand(next, qPrintable(move_side))) {
                return false;
            }
        }
    } while (!stop && passes < 2 && move_num < (19 * 19 * 2));

    cerr << endl;

    // Nobody resigned, we will have to count
    if (!stop) {
        // Ask for the winner
        first_process.write(qPrintable("final_score\n"));
        first_process.waitForBytesWritten(-1);
        if (!waitForReadyRead(first_process)) {
            return false;
        }
        read_cnt = first_process.readLine(readbuff, 256);
        QString score(&readbuff[2]);
        cerr << "Score: " << score;
        // final_score returns
        // "= W+" or "= B+"
        if (readbuff[2] == 'W') {
            winner = QString(QStringLiteral("white"));
        } else if (readbuff[2] == 'B') {
            winner = QString(QStringLiteral("black"));
        }
        cerr << "Winner: " << winner << endl;
        // Double newline
        if (!waitForReadyRead(first_process)) {
            return false;
        }
        read_cnt = first_process.readLine(readbuff, 256);
        Q_ASSERT(read_cnt > 0);
    } else {
        if (black_resigned) {
            winner = QString(QStringLiteral("white"));
        } else {
            winner = QString(QStringLiteral("black"));
        }
    }

    if (winner.isNull()) {
        cerr << "No winner found" << endl;
        first_process.write(qPrintable("quit\n"));
        second_process.write(qPrintable("quit\n"));

        first_process.waitForFinished(-1);
        second_process.waitForFinished(-1);
        return false;
    }

    // Write the game SGF
    QString sgf_name, training_name;
    QString random_name(QUuid::createUuid().toRfc4122().toHex());

    sgf_name += random_name;
    sgf_name += ".sgf";
    training_name += random_name;
    training_name += ".txt";

    cerr << "Writing " << sgf_name << endl;

    if (!sendGtpCommand(first_process,
                        qPrintable("printsgf " + sgf_name + "\n"))) {
        return false;
    }

    QString dump_cmd(qPrintable("dump_training " + winner +
                     " " + training_name + "\n"));
    cerr << dump_cmd;

    // Now dump the training
    if (!sendGtpCommand(first_process, dump_cmd)) {
        return false;
    }
    if (!sendGtpCommand(second_process, dump_cmd)) {
        return false;
    }

    // Close down
    first_process.write(qPrintable("quit\n"));
    second_process.write(qPrintable("quit\n"));

    first_process.waitForFinished(-1);
    second_process.waitForFinished(-1);

    return true;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QTimer::singleShot(0, &app, SLOT(quit()));

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

    cerr << "autogtp v0.1" << endl;

    auto success = true;
    auto games_played = 0;

    do {
        QString netname;
        success &= fetch_best_network_hash(cerr, netname);
        success &= fetch_best_network(cerr, netname);
        success &= run_one_game(cerr, netname);
        success &= upload_data(cerr, netname);
        games_played++;
        cerr << games_played << " games played." << endl;
    } while (success);

    cerr.flush();
    cout.flush();
    return app.exec();
}
