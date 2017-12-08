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

#include <cmath>
#include <random>
#include <QDir>
#include <QFileInfo>
#include <QThread>
#include <QCryptographicHash>
#include <QJsonDocument>
#include <QJsonObject>
#include "Management.h"
#include "Game.h"

constexpr int RETRY_DELAY_MIN_SEC = 30;
constexpr int RETRY_DELAY_MAX_SEC = 60 * 60;  // 1 hour
constexpr int MAX_RETRIES = 4 * 24;           // Stop retrying after 4 days

Management::Management(const int gpus,
                       const int games,
                       const QStringList& gpuslist,
                       const int ver,
                       const QString& keep,
                       const QString& debug,
                       QMutex* mutex)
    : m_mainMutex(mutex),
    m_syncMutex(),
    m_gamesThreads(gpus * games),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_selfGames(0),
    m_matchGames(0),
    m_gamesPlayed(0),
    m_keepPath(keep),
    m_debugPath(debug),
    m_version(ver) {
}

void Management::giveAssignments() {
    m_start = std::chrono::high_resolution_clock::now();
    m_mainMutex->lock();
    QString myGpu;
    for (int gpu = 0; gpu < m_gpus; ++gpu) {
        for (int game = 0; game < m_games; ++game) {
            int thread_index = gpu * m_games + game;
            if (m_gpusList.isEmpty()) {
                myGpu = "";
            } else {
                myGpu = m_gpusList.at(gpu);
            }
            m_gamesThreads[thread_index] = new Worker(thread_index, myGpu, m_keepPath);
            connect(m_gamesThreads[thread_index],
                    &Worker::resultReady,
                    this,
                    &Management::getResult,
                    Qt::DirectConnection);
            m_gamesThreads[thread_index]->order(getWork());
            m_gamesThreads[thread_index]->start();
        }
    }
}

void Management::getResult(Order ord, Result res, int index, int duration) {
    if (res.type() == Result::Error) {
        exit(1);
    }
    m_syncMutex.lock();
    m_gamesPlayed++;
    switch(res.type()) {
    case Result::File:
        m_selfGames++,
        m_movesMade += res.list()[1].toInt();
        uploadData(res.list()[0], ord.parameters()[3], ord.parameters()[2]);
        break;
    case Result::Win:
    case Result::Loss:
        m_matchGames++,
        m_movesMade += res.list()[4].toInt();
        uploadResult(res.list(), ord.parameters());
        break;
    }
    printTimingInfo(duration);
    m_gamesThreads[index]->order(getWork());
    m_syncMutex.unlock();

}


void  Management::printTimingInfo(float duration) {

    auto game_end = std::chrono::high_resolution_clock::now();
    auto total_time_s =
        std::chrono::duration_cast<std::chrono::seconds>(game_end - m_start);
    auto total_time_min =
        std::chrono::duration_cast<std::chrono::minutes>(total_time_s);
    auto total_time_millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_time_s);
    QTextStream(stdout)
        << m_gamesPlayed << " game(s) (" << m_selfGames << " self played and "
        << m_matchGames << " matches) played in "
        << total_time_min.count() << " minutes = "
        << total_time_s.count() / m_gamesPlayed << " seconds/game, "
        << total_time_millis.count() / m_movesMade  << " ms/move"
        << ", last game took " << (int) duration << " seconds." << endl;
}

QString Management::getOption(const QJsonObject &ob, const QString &key, const QString &opt, const QString &defValue) {
    QString res;
    if (ob.contains(key)) {
        res.append(opt + ob.value(key).toString() + " ");
    } else {
        res.append(opt + defValue + " ");
    }
    return res;
}

QString Management::getBoolOption(const QJsonObject &ob, const QString &key, const QString &opt, bool defValue) {
    QString res;
    if (ob.contains(key) && ob.value(key).toString().compare("true", Qt::CaseInsensitive)) {
        res.append(opt + " ");
    } else {
        if(defValue) {
            res.append(opt + " ");
        }
    }
    return res;
}


Order Management::getWork() {
    Order o;
    o.type(Order::Error);

    /*

{
   "cmd" : "match",
   "white_hash" : "223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6",
   "black_hash" : "92c658d7325fe38f0c8adbbb1444ed17afd891b9f208003c272547a7bcb87909",
   "options_hash" : "c2e3"
   "required_client_version" : "5",
   "options" : {
       "playouts" : "1000",
       "resignation_percent" : "3",
       "noise" : "false",
       "randomcnt" : "0"
    }
}

{
   "cmd" : "selfplay",
   "hash" : "223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6",
   "options_hash" : "ee21",
   "required_client_version" : "5",
   "options" : {
       "playouts" : 1000,
       "resignation_percent" : "3",
       "noise" : "true",
       "randomcnt" : "30"
    }
}
    */
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    prog_cmdline.append(" -s -J");
    prog_cmdline.append(" http://zero-test.sjeng.org/get-task/7");

    QTextStream(stdout) << prog_cmdline << endl;

    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    if (curl.exitCode()) {
        throw NetworkException("Curl returned non-zero exit code "
                               + std::to_string(curl.exitCode()));
        return o;
    }
    QJsonDocument doc;
    doc = QJsonDocument::fromJson(curl.readAllStandardOutput());
    QTextStream(stdout) << doc.toJson() << endl;
    QJsonObject ob = doc.object();
    QJsonObject opt = ob.value("options").toObject();
    QString options;
    QString optionsHash =  ob.value("options_hash").toString();
    if (ob.contains("required_client_version")) {
        QTextStream(stdout) << "Required client version: " << ob.value("required_client_version").toString() << endl;
        if (ob.value("required_client_version").toString().toInt() > m_version) {
            QTextStream(stdout) << ' ' <<  endl;
            QTextStream(stdout)
                << "Server requires client version " << ob.value("required_client_version").toString()
                << " but we are version " << m_version << endl;
            QTextStream(stdout)
                << "Check https://github.com/gcp/leela-zero for updates." << endl;
            exit(EXIT_FAILURE);
        }
    }
    QString leelazVersion = "0.8";
    if (ob.contains("leelaz_version")) {
        leelazVersion = ob.value("leelaz_version").toString();
    }
    options.append(getOption(opt, "playouts", " -p ", "1600"));
    options.append(getOption(opt, "resignation_percent", " -r ", "1"));
    options.append(getOption(opt, "randomcnt", " -m ", "30"));
    options.append(getOption(opt, "threads", " -t ", "1"));
    options.append(getBoolOption(opt, "dumbpass", " -d ", true));
    options.append(getBoolOption(opt, "noise", " -n ", true));
    options.append(" --noponder ");
    QStringList parameters;
    QTextStream(stdout) << options << endl;
    parameters << leelazVersion;   //[0]
    parameters << options;         //[1]
    parameters << optionsHash;     //[2]
    if (ob.value("cmd").toString() == "selfplay") {
        QString net = ob.value("hash").toString();
        fetchNetwork(net);
        o.type(Order::Production);
        o.parameters(parameters << net);
    }
    if (ob.value("cmd").toString() == "match") {
        o.type(Order::Validation);
        QString net1 = ob.value("black_hash").toString();
        QString net2 = ob.value("white_hash").toString();
        fetchNetwork(net1);
        fetchNetwork(net2);
        o.parameters(parameters << net1 << net2);
    }
    return o;
}


bool Management::networkExists(const QString &name) {
    if (QFileInfo::exists(name)) {
        QFile f(name);
        if (f.open(QFile::ReadOnly)) {
            QCryptographicHash hash(QCryptographicHash::Sha256);
            if (!hash.addData(&f)) {
                QTextStream(stdout) << "Reading network file failed." << endl;
                exit(EXIT_FAILURE);
            }
            QString result = hash.result().toHex();
            if (result == name) {
                return true;
            }
        } else {
            QTextStream(stdout)
                << "Unable to open network file for reading." << endl;
            if (f.remove()) {
                return false;
            }
            QTextStream(stdout) << "Unable to delete the network file. "
                                << "Check permissions." << endl;
            exit(EXIT_FAILURE);
        }
        QTextStream(stdout) << "Downloaded network hash doesn't match." << endl;
        f.remove();
    }
    return false;
}

void Management::fetchNetwork(const QString &name) {
    if (networkExists(name)) {
        QTextStream(stdout) << "Already downloaded network." << endl;
        return;
    }
    if (QFileInfo::exists(name + ".gz")) {
        QFile f_gz(name + ".gz");
        // Curl refuses to overwrite, so make sure to delete the gzipped
        // network if it exists
        f_gz.remove();
    }

    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    // Be quiet, but output the real file name we saved.
    // Use the filename from the server.
    prog_cmdline.append(" -s -O -J");
    prog_cmdline.append(" -w %{filename_effective}");
    prog_cmdline.append(" http://zero.sjeng.org/networks/" + name + ".gz");

    QTextStream(stdout) << prog_cmdline << endl;

    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    if (curl.exitCode()) {
        throw NetworkException("Curl returned non-zero exit code "
                               + std::to_string(curl.exitCode()));
    }

    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QStringList outlst = outstr.split("\n");
    QString outfile = outlst[0];
    QTextStream(stdout) << "Curl filename: " << outfile << endl;
#ifdef WIN32
    QProcess::execute("gzip.exe -d -q " + outfile);
#else
    QProcess::execute("gunzip -q " + outfile);
#endif
    // Remove extension (.gz)
    outfile.chop(3);
    QTextStream(stdout) << "Net filename: " << outfile << endl;

    if (!networkExists(name)) {
        exit(EXIT_FAILURE);
    }

    return;
}

/*
-F winnerhash=223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6
-F loserhash=92c658d7325fe38f0c8adbbb1444ed17afd891b9f208003c272547a7bcb87909
-F clientversion=6
-F winnercolor=black
-F movescount=321
-F score=B+45
-F options_hash=c2e3
-F sgf=@file
http://zero-test.sjeng.org/submit-match
*/

void Management::uploadResult(const QStringList &r, const QStringList &l) {

    QString gzipCmd ="gzip";
#ifdef WIN32
    gzipCmd.append(".exe");
#endif
    gzipCmd.append(" " + r[3] + ".sgf");
    QProcess::execute(gzipCmd);
    QString sgf_file = r[3] + ".sgf.gz";
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    if (r[2] == "black") {
        prog_cmdline.append(" -F winnerhash=" + l[3]);
        prog_cmdline.append(" -F loserhash=" + l[4]);
    } else {
        prog_cmdline.append(" -F winnerhash=" + l[4]);
        prog_cmdline.append(" -F loserhash=" + l[3]);
    }
    prog_cmdline.append(" -F clientversion=" + QString::number(m_version));
    prog_cmdline.append(" -F winnercolor="+ r[2]);
    prog_cmdline.append(" -F movescount="+ r[0]);
    prog_cmdline.append(" -F score="+ r[1]);
    prog_cmdline.append(" -F options_hash="+ l[2]);
    prog_cmdline.append(" -F sgf=@"+ sgf_file);
    prog_cmdline.append(" http://zero-test.sjeng.org/submit-match");

    QTextStream(stdout) << prog_cmdline << endl;
    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    if (curl.exitCode()) {
        QTextStream(stdout) << "Upload failed. Curl Exit code: "
            << curl.exitCode() << endl;
    }
    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QTextStream(stdout) << outstr;
    QDir dir;
    dir.remove(sgf_file);
}


/*
-F networkhash=223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6
-F clientversion=6
-F options_hash=ee21
-F sgf=@file
-F trainingdata=@data_file
http://zero-test.sjeng.org/submit
*/

void Management::uploadData(const QString& file, const QString& net , const QString &hash) {
    // Find output SGF and txt files
    QTextStream(stdout) << "Upload game: " << file << " network " << net << endl;
    QDir dir;
    QStringList filters;
    filters << file + ".sgf";
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);

    QFileInfoList list = dir.entryInfoList();
    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fileInfo = list.at(i);
        QString sgf_file = fileInfo.fileName();
        QString data_file = sgf_file;
        // Cut .sgf, add .txt.0.gz
        data_file.chop(4);
        QString debug_data_file = data_file;
        data_file += ".txt.0.gz";
        debug_data_file += ".txt.debug.0.gz";
        // Save first if requested
        if (!m_keepPath.isEmpty()) {
            QFile(sgf_file).copy(m_keepPath + '/' + sgf_file);
        }
        if (!m_debugPath.isEmpty()) {
            QFile(data_file).copy(m_debugPath + '/' + data_file);
            QFile(debug_data_file).copy(m_debugPath + '/' + debug_data_file);
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
        prog_cmdline.append(" -F networkhash=" + net);
        prog_cmdline.append(" -F clientversion=" + QString::number(m_version));
        prog_cmdline.append(" -F options_hash="+ hash);
        prog_cmdline.append(" -F sgf=@" + sgf_file);
        prog_cmdline.append(" -F trainingdata=@" + data_file);
        prog_cmdline.append(" http://zero-test.sjeng.org/submit");
        QTextStream(stdout) << prog_cmdline << endl;
        QProcess curl;
        curl.start(prog_cmdline);
        curl.waitForFinished(-1);

        if (curl.exitCode()) {
            QTextStream(stdout) << "Upload failed. Curl Exit code: "
                << curl.exitCode() << endl;
            QTextStream(stdout) << "Continuing..." << endl;
        }

        QByteArray output = curl.readAllStandardOutput();
        QString outstr(output);
        QTextStream(stdout) << outstr;
        dir.remove(sgf_file);
        dir.remove(data_file);
        dir.remove(debug_data_file);
    }
    return;
}
