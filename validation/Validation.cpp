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

using VersionTuple = std::tuple<int, int>;
// Minimal Leela Zero version we expect to see
const VersionTuple min_leelaz_version{0, 9};


void ValidationWorker::run() {
    do {
        Game first(m_firstNet,  m_option);
        if (!first.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, Game::BLACK);
            return;
        }
        Game second(m_secondNet, m_option);
        if (!second.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, Game::BLACK);
            return;
        }
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
            if (m_expected == Game::BLACK) {
                m_expected = Game::WHITE;
            } else {
                m_expected = Game::BLACK;
            }
        }
    } while (m_state.load() != FINISHING);
}

void ValidationWorker::init(const QString& gpuIndex,
                            const QString& firstNet,
                            const QString& secondNet,
                            const QString& keep,
                            const int expected) {
    m_option = " -g  -p 1600 --noponder -t 1 -q -d -r 0 -w ";
    if (!gpuIndex.isEmpty()) {
        m_option.prepend(" --gpu=" + gpuIndex + " ");
    }
    m_firstNet = firstNet;
    m_secondNet = secondNet;
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
                       QMutex* mutex) :
    m_mainMutex(mutex),
    m_syncMutex(),
    m_gamesThreads(gpus*games),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_firstNet(firstNet),
    m_secondNet(secondNet),
    m_keepPath(keep) {
    m_statistic.initialize(0.0, 35.0, 0.05, 0.05);
    m_statistic.addGameResult(Sprt::Draw);
}

void Validation::startGames() {
    m_mainMutex->lock();
    QString n1, n2;
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
                expected = Game::BLACK;
            } else {
                n1 = m_secondNet;
                n2 = m_firstNet;
                expected = Game::WHITE;
            }
            if (m_gpusList.isEmpty()) {
                myGpu = "";
            } else {
                myGpu = m_gpusList.at(gpu);
            }
            m_gamesThreads[thread_index].init(myGpu, n1, n2, m_keepPath, expected);
            m_gamesThreads[thread_index].start();
        }
    }
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
        m_mainMutex->unlock();
    } else {
        QTextStream(stdout)
            << m_results.getGamesPlayed() << " games played." << endl;
        QTextStream(stdout)
            << "Status: " << status.result
            << " LLR " << status.llr
            << " Lower Bound " << status.lBound
            << " Upper Bound " << status.uBound << endl;
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
