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

#include <QDir>
#include <QFileInfo>
#include "Production.h"
#include "Game.h"

void ProductionWorker::run(){
     do {
         std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
         Game game(m_network, m_option);
         if(!game.gameStart(min_leelaz_version)) {
             return;
         }
         do {
             game.move();
             if(!game.waitForMove()) {
                 return;
             }
             game.readMove();
         } while (game.nextMove());
         QTextStream(stdout) << "Game has ended." << endl;
         if (game.getScore()) {
             game.writeSgf();
             game.dumpTraining();
         }
         QTextStream(stdout) << "Stopping engine." << endl;
         game.gameQuit();
         std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
         float gameDuration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
         emit resultReady(game.getFile(), gameDuration);
    } while (1);
}

void ProductionWorker::init(const QString &gpuIndex, const QString &net) {
    m_option = " -g -q -n -d -m 30 -r 0 -w ";
    if(!gpuIndex.isEmpty())
        m_option.prepend(" -gpu=" + gpuIndex + " ");
    m_network = net;
}


Production::Production(const int &gpus, const int &games, const QStringList &gpuslist, const int &ver, const QString &keep, QMutex *mutex) :
    m_mainMutex(mutex),
    m_syncMutex(),
    m_gamesThreads(gpus*games),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_gamesPlayed(0),
    m_keepPath(keep),
    m_version(ver)
{
}

Production::~Production() {}

void Production::startGames() {
    m_start = std::chrono::high_resolution_clock::now();
    m_mainMutex->lock();
    fetchBestNetworkHash();
    fetchBestNetwork();
    QString myGpu;
    for(int gpu = 0; gpu < m_gpus; ++gpu) {
        for(int game = 0; game < m_games; ++game) {
            connect(&m_gamesThreads[gpu * m_games + game], &ProductionWorker::resultReady, this, &Production::getResult, Qt::DirectConnection);
            if(m_gpusList.isEmpty()) {
                myGpu = "";
            }
            else {
                myGpu = m_gpusList.at(gpu);
            }
            m_gamesThreads[gpu * m_games + game].init(myGpu, m_network);
            m_gamesThreads[gpu * m_games + game].start();
        }
    }
}

void Production::getResult(QString file, float duration) {
    m_syncMutex.lock();
    m_gamesPlayed++;
    printTimingInfo(duration);
    uploadData(file);
    if(!fetchBestNetworkHash()) {
        fetchBestNetwork();
        ProductionWorker *w = qobject_cast<ProductionWorker*>(sender());
        w->newNetwork(m_network);
    }
    m_syncMutex.unlock();
}

void  Production::printTimingInfo(float duration) {
    auto game_end = std::chrono::high_resolution_clock::now();
    auto total_time_s = std::chrono::duration_cast<std::chrono::seconds>(game_end - m_start);
    auto total_time_min = std::chrono::duration_cast<std::chrono::minutes>(total_time_s);
    QTextStream(stdout) << m_gamesPlayed << " game(s) played in "
         << total_time_min.count() << " minutes = "
         << total_time_s.count() / m_gamesPlayed << " seconds/game"
         << ", last game took "
         << duration << " seconds.\n";
}


bool Production::fetchBestNetworkHash() {
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
        QTextStream(stdout) << "Unexpected output from server: " << endl << output << endl;
        exit(EXIT_FAILURE);
    }
    QString outhash = outlst[0];
    QTextStream(stdout) << "Best network hash: " << outhash << endl;
    QString client_version = outlst[1];
    auto server_expected = client_version.toInt();
    QTextStream(stdout) << "Required client version: " << client_version;
    if (server_expected > m_version) {
        QTextStream(stdout) << ' ' <<  endl;
        QTextStream(stdout) << "Server requires client version " << server_expected
                            << " but we are version " << m_version << endl;
        QTextStream(stdout) << "Check https://github.com/gcp/leela-zero for updates." << endl;
        exit(EXIT_FAILURE);
    } else {
        QTextStream(stdout) << " (OK)" << endl;
    }
    if(outhash == m_network) {
        return true;
    }
    m_network = outhash;
    return false;

}

void Production::fetchBestNetwork() {
    if (QFileInfo::exists(m_network)) {
        QTextStream(stdout) << "Already downloaded network." << endl;
        return;
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

    QTextStream(stdout) << prog_cmdline << endl;

    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QStringList outlst = outstr.split("\n");
    QString outfile = outlst[0];
    QTextStream(stdout) << "Curl filename: " << outfile << endl;
#ifdef WIN32
    QProcess::execute("gzip.exe -d -k -q " + outfile);
#else
    QProcess::execute("gunzip -k -q " + outfile);
#endif
    // Remove extension (.gz)
    outfile.chop(3);
    QTextStream(stdout) << "Net filename: " << outfile << endl;
    m_network = outfile;
    return;
}

void Production::uploadData(const QString &file) {
    // Find output SGF and txt files
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
        // Save first if requested
        if (!m_keepPath.isEmpty()) {
            QFile(sgf_file).copy(m_keepPath + '/' + fileInfo.fileName());
        }
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
        prog_cmdline.append(" -F networkhash=" + m_network);
        prog_cmdline.append(" -F clientversion=" + QString::number(m_version));
        prog_cmdline.append(" -F sgf=@" + sgf_file);
        prog_cmdline.append(" -F trainingdata=@" + data_file);
        prog_cmdline.append(" http://zero.sjeng.org/submit");
        QTextStream(stdout) << prog_cmdline << endl;
        QProcess curl;
        curl.start(prog_cmdline);
        curl.waitForFinished(-1);
        QByteArray output = curl.readAllStandardOutput();
        QString outstr(output);
        QTextStream(stdout) << outstr;
        dir.remove(sgf_file);
        dir.remove(data_file);
    }
    return;
}
