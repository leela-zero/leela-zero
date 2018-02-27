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

#include "Validation.h"
#include <QFile>
#include <QDir>
#include <QUuid>

using VersionTuple = std::tuple<int, int, int>;
// Minimal Leela Zero version we expect to see
const VersionTuple min_leelaz_version{0, 10, 0};


void ValidationWorker::run() {
    do {
        Game first(m_firstNet,  m_firstOpts, m_firstBin);
        if (!first.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, Game::BLACK);
            return;
        }
        Game second(m_secondNet, m_secondOpts, m_secondBin);
        if (!second.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, Game::BLACK);
            return;
        }
        QTextStream(stdout) << "starting:" << endl <<
            first.getCmdLine() << endl <<
            "vs" << endl <<
            second.getCmdLine() << endl;
            
        QString wmove = "play white ";
        QString bmove = "play black ";
        do {
            first.move();
            if(!first.waitForMove()) {
                emit resultReady(Sprt::NoResult, Game::BLACK);
                return;
            }
            first.readMove();
            if(first.checkGameEnd()) {
                break;
            }
            second.setMove(bmove + first.getMove());
            second.move();
            if(!second.waitForMove()) {
                emit resultReady(Sprt::NoResult, Game::BLACK);
                return;
            }
            second.readMove();
            first.setMove(wmove + second.getMove());
            second.nextMove();
        } while (first.nextMove() && m_state.load() == RUNNING);

        if (m_state.load() == RUNNING) {
            QTextStream(stdout) << "Game has ended." << endl;
            int result = 0;
            if (first.getScore()) {
                result = first.getWinner();
                if (!m_keepPath.isEmpty()) {
                    first.writeSgf();
                    QString prefix = m_keepPath + '/';
                    if(m_expected == Game::BLACK) {
                        prefix.append("black_");
                    } else {
                        prefix.append("white_");
                    }
                    QFile(first.getFile() + ".sgf").rename(prefix + first.getFile() + ".sgf");
                }
            }
            QTextStream(stdout) << "Stopping engine." << endl;
            first.gameQuit();
            second.gameQuit();

            // Game is finished, send the result
            if (result == m_expected) {
                emit resultReady(Sprt::Win, m_expected);
            } else {
                emit resultReady(Sprt::Loss, m_expected);
            }
            // Change color and play again
            m_firstNet.swap(m_secondNet);
            m_firstBin.swap(m_secondBin);
            m_firstOpts.swap(m_secondOpts);
            if (m_expected == Game::BLACK) {
                m_expected = Game::WHITE;
            } else {
                m_expected = Game::BLACK;
            }
        } else {
            first.gameQuit();
            second.gameQuit();
        }
    } while (m_state.load() != FINISHING);
}

void ValidationWorker::init(const QString& gpuIndex,
                            const QString& firstNet,
                            const QString& secondNet,
                            const QString& firstBin,
                            const QString& secondBin,
                            const QString& firstOpts,
                            const QString& secondOpts,
                            const QString& keep,
                            int expected) {
    m_firstOpts = firstOpts;
    m_secondOpts = secondOpts;
    if (!gpuIndex.isEmpty()) {
        m_firstOpts.prepend(" --gpu=" + gpuIndex + " ");
        m_secondOpts.prepend(" --gpu=" + gpuIndex + " ");
    }
    m_firstNet = firstNet;
    m_secondNet = secondNet;
    m_firstBin = firstBin;
    m_secondBin = secondBin;
    m_expected = expected;
    m_keepPath = keep;
    m_state.store(RUNNING);
}

Validation::Validation(const int gpus,
                       const int games,
                       const QStringList& gpuslist,
                       const QString& firstNet,
                       const QString& secondNet,
                       const QString& keep,
                       QMutex* mutex,
                       const QString& firstBin,
                       const QString& secondBin,
                       const QString& firstOpts,
                       const QString& secondOpts,
                       const float& h0,
                       const float& h1) :
        
    m_mainMutex(mutex),
    m_syncMutex(),
    m_gamesThreads(gpus*games),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_firstNet(firstNet),
    m_secondNet(secondNet),
    m_firstBin(firstBin),
    m_secondBin(secondBin),
    m_firstOpts(firstOpts),
    m_secondOpts(secondOpts),
    m_keepPath(keep) {
    m_statistic.initialize(h0, h1, 0.05, 0.05);
    m_statistic.addGameResult(Sprt::Draw);
}

void Validation::startGames() {
    QString n1, n2, b1 ,b2 ,o1, o2;
    int expected;
    QString myGpu;
    for(int gpu = 0; gpu < m_gpus; ++gpu) {
        for(int game = 0; game < m_games; ++game) {
            auto thread_index = gpu * m_games + game;
            connect(&m_gamesThreads[thread_index],
                    &ValidationWorker::resultReady,
                    this,
                    &Validation::getResult,
                    Qt::DirectConnection);
            if (game % 2) {
                n1 = m_firstNet;
                n2 = m_secondNet;
                b1 = m_firstBin;
                b2 = m_secondBin;
                o1 = m_firstOpts;
                o2 = m_secondOpts;
                expected = Game::BLACK;
            } else {
                n1 = m_secondNet;
                n2 = m_firstNet;
                b1 = m_secondBin;
                b2 = m_firstBin;
                o1 = m_secondOpts;
                o2 = m_firstOpts;
                expected = Game::WHITE;
            }
            if (m_gpusList.isEmpty()) {
                myGpu = "";
            } else {
                myGpu = m_gpusList.at(gpu);
            }

            m_gamesThreads[thread_index].init(myGpu, n1, n2, b1, b2, o1, o2, m_keepPath, expected);
            m_gamesThreads[thread_index].start();
        }
    }
}

void Validation::saveSprt() {
    QFile f("sprtsave" + QUuid::createUuid().toRfc4122().toHex() + ".bin");
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return;
    }
    QTextStream out(&f);
    out << m_statistic;
    out << m_results;
    f.close();
    m_results.printResults(m_firstNet, m_secondNet);
    printSprtStatus(m_statistic.status());
}

void Validation::loadSprt() {
    QDir dir;
    QStringList filters;
    filters << "sprtsave*.bin";
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    if (dir.entryInfoList().isEmpty()) {
        return;
    }
    QFileInfo fi = dir.entryInfoList().takeFirst();
    QFile f(fi.fileName());
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return;
    }
    QTextStream in(&f);
    in >> m_statistic;
    in >> m_results;
    f.close();
    QFile::remove(fi.fileName());
    QTextStream(stdout) << "Initial Statistics" << endl;
    m_results.printResults(m_firstNet, m_secondNet);
    printSprtStatus(m_statistic.status());
}

void Validation::printSprtStatus(const Sprt::Status& status) {
    QTextStream(stdout)
        << m_results.getGamesPlayed() << " games played." << endl;
    QTextStream(stdout)
        << "Status: " << status.result
        << " LLR " << status.llr
        << " Lower Bound " << status.lBound
        << " Upper Bound " << status.uBound << endl;
}

void Validation::getResult(Sprt::GameResult result, int net_one_color) {
    if(result == Sprt::NoResult) {
        QTextStream(stdout) << "Engine Error." << endl;
        return;
    }
    m_syncMutex.lock();
    m_statistic.addGameResult(result);
    m_results.addGameResult(result, net_one_color);

    Sprt::Status status = m_statistic.status();
    auto wdl = m_statistic.getWDL();
    QTextStream(stdout) << std::get<0>(wdl) << " wins, "
                        << std::get<2>(wdl) << " losses" << endl;
    if(status.result != Sprt::Continue) {
        quitThreads();
        QTextStream(stdout)
            << "The first net is "
            <<  ((status.result ==  Sprt::AcceptH0) ? "worse " : "better ")
            << "than the second" << endl;
        m_results.printResults(m_firstNet, m_secondNet);
        //sendQuit();
    } else {
        printSprtStatus(status);
    }
    m_syncMutex.unlock();
}

void Validation::quitThreads() {
    for(int gpu = 0; gpu < m_gpus * m_games; ++gpu) {
        m_gamesThreads[gpu].doFinish();
    }
}

void Validation::wait() {
    for(int gpu = 0; gpu < m_gpus * m_games; ++gpu) {
        m_gamesThreads[gpu].wait();
    }
}

void Validation::storeSprt() {
    QTextStream(stdout) << "storeSprt" << endl;
    m_syncMutex.lock();
    saveSprt();
    m_syncMutex.unlock();
    quitThreads();
    wait();
}
