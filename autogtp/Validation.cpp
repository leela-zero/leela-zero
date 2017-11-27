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
#include "Game.h"
#include <QFile>

void ValidationWorker::run() {
    // TODO delete before code is merged, useful for testing output.
    while (1) {
        sleep(1);

        Sprt::GameResult result = (rand() % 10 <= 8) ? Sprt::Win : Sprt::Loss;
        int net_one_color = rand() % 2 ? Game::BLACK : Game::WHITE;
        emit resultReady(result, net_one_color);
    }

    do {
        Game first(m_firstNet,  m_option);
        if(!first.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, Game::BLACK);
            return;
        }
        Game second(m_secondNet, m_option);
        if(!second.gameStart(min_leelaz_version)) {
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
            second.setMove(bmove + first.getMove());
            second.move();
            if(!second.waitForMove()) {
                emit resultReady(Sprt::NoResult, Game::BLACK);
                return;
            }
            second.readMove();
            first.setMove(wmove + second.getMove());
            second.nextMove();
        } while (first.nextMove());
        QTextStream(stdout) << "Game has ended." << endl;
        int result = 0;
        if (first.getScore()) {
            result = first.getWinner();
        }
        if(!m_keepPath.isEmpty())
        {
            first.writeSgf();
            QString prefix = m_keepPath + '/';
            if(m_expected == Game::BLACK) {
                prefix.append("black_");
            } else {
                prefix.append("white_");
            }
            QFile(first.getFile() + ".sgf").rename(prefix + first.getFile() + ".sgf");
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
        QString net;
        net = m_secondNet;
        m_secondNet = m_firstNet;
        m_firstNet = net;
        if(m_expected == Game::BLACK) {
            m_expected = Game::WHITE;
        } else {
            m_expected = Game::BLACK;
        }
    } while (1);
}

void ValidationWorker::init(const QString& gpuIndex,
                            const QString& firstNet,
                            const QString& secondNet,
                            const QString& keep,
                            const int expected) {
    m_option = " -g -q -r 0 -w ";
    if (!gpuIndex.isEmpty()) {
        m_option.prepend(" --gpu=" + gpuIndex + " ");
    }
    m_firstNet = firstNet;
    m_secondNet = secondNet;
    m_expected = expected;
    m_keepPath = keep;
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
    m_gamesPlayed(0),
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
    m_gamesPlayed++;
    m_statistic.addGameResult(result);
    if (net_one_color == Game::BLACK) {
        m_blackStatistic.addGameResult(result);
    } else {
        m_whiteStatistic.addGameResult(result);
    }

    Sprt::Status status = m_statistic.status();
    auto wdl = m_statistic.getWDL();
    QTextStream(stdout) << std::get<0>(wdl) << " wins, "
                        << std::get<2>(wdl) << " losses" << endl;
    if(status.result != Sprt::Continue) {
        quitThreads();
        QTextStream(stdout) << "The first net is ";
        QTextStream(stdout)
            <<  ((status.result ==  Sprt::AcceptH0) ? "worse " : "better ");
        QTextStream(stdout) << "than the second" << endl;
        printResult();
        m_mainMutex->unlock();
    }
    else {
        QTextStream(stdout) << m_gamesPlayed << " games played." << endl;
        QTextStream(stdout)
            << "Status: " << status.result << " LLR ";
        QTextStream(stdout) << status.llr <<  " Lower Bound " << status.lBound;
        QTextStream(stdout) << " Upper Bound " << status.uBound << endl;
    }
    m_syncMutex.unlock();
}


void Validation::printResult() {
        /*
        Produces reports in this format
        leelaz-9k v leelaz-19k (176 games)
                     wins              black         white
        leelaz-9k      65 36.93%       37 42.53%     28 31.46%
        leelaz-19k    111 63.07%       61 68.54%     50 57.47%
                                       98 55.68%     78 44.32%
        */

        QString first_name = "leelaz-" + m_firstNet.left(8);
        QString second_name = "leelaz-" + m_secondNet.left(8);
        QTextStream(stdout) << "\n" << first_name << " v " << second_name
                            << " (" << m_gamesPlayed << " games)" << endl;

        auto wdl = m_statistic.getWDL();
        auto black_wdl = m_blackStatistic.getWDL();
        auto black_games = std::get<0>(black_wdl) + std::get<2>(black_wdl);
        auto white_wdl = m_whiteStatistic.getWDL();
        auto white_games = std::get<0>(white_wdl) + std::get<2>(white_wdl);
        auto sum_black_wins = std::get<0>(black_wdl) + std::get<2>(white_wdl);
        auto sum_white_wins = std::get<2>(black_wdl) + std::get<0>(white_wdl);

        // Using sprintf, a little more compact
        char report[4][100];
        sprintf(report[0], "%-21s %-14s %-14s %s\n",
            "" /* name */, "wins", "black", "white");
        sprintf(report[1], "%-16s %4d %5.2f%% %7d %5.2f%% %7d %5.2f%%\n",
            first_name.toLocal8Bit().constData(),
            std::get<0>(wdl), 100.0*std::get<0>(wdl)/m_gamesPlayed,
            std::get<0>(black_wdl), 100.0*std::get<0>(black_wdl)/black_games,
            std::get<0>(white_wdl), 100.0*std::get<0>(white_wdl)/white_games);
        sprintf(report[2], "%-16s %4d %5.2f%% %7d %5.2f%% %7d %5.2f%%\n",
            second_name.toLocal8Bit().constData(),
            std::get<2>(wdl), 100.0*std::get<2>(wdl)/m_gamesPlayed,
            std::get<2>(black_wdl), 100.0*std::get<2>(black_wdl)/black_games,
            std::get<2>(white_wdl), 100.0*std::get<2>(white_wdl)/white_games);
        sprintf(report[3], "%16s %14s %4d %5.2f%% %7d %5.2f%%",
            "" /* name */, "" /* wins column */,
            sum_black_wins, 100.0*sum_black_wins/m_gamesPlayed,
            sum_white_wins, 100.0*sum_white_wins/m_gamesPlayed);
        QTextStream(stdout)
            << report[0] << report[1] << report[2] << report[3] << endl;

        QTextStream(stdout) << "\n";

        // Using QString arg, more QT esque
        QTextStream(stdout) << QString("%1 %2 %3 %4\n")
            .arg("" /* name */, 16)
            .arg("wins", -16)
            .arg("black", -14)
            .arg("white");
        QTextStream(stdout) << QString("%1 %2 %3\% %4 %5\% %6 %7\%\n")
            .arg(first_name, -16)
            .arg(std::get<0>(wdl), 4)
            .arg(100.0 * std::get<0>(wdl) / m_gamesPlayed, 5, 'g', 4)
            .arg(std::get<0>(black_wdl), 7)
            .arg(100.0 * std::get<0>(black_wdl) / black_games, 5, 'g', 4)
            .arg(std::get<0>(white_wdl), 7)
            .arg(100.0 * std::get<0>(white_wdl) / white_games, 5, 'g', 4);
        QTextStream(stdout) << QString("%1 %2 %3\% %4 %5\% %6 %7\%\n")
            .arg(second_name, -16)
            .arg(std::get<2>(wdl), 4)
            .arg(100.0 * std::get<2>(wdl) / m_gamesPlayed, 5, 'g', 4)
            .arg(std::get<2>(black_wdl), 7)
            .arg(100.0 * std::get<2>(black_wdl) / black_games, 5, 'g', 4)
            .arg(std::get<2>(white_wdl), 7)
            .arg(100.0 * std::get<2>(white_wdl) / white_games, 5, 'g', 4);
        QTextStream(stdout) << QString("%1 %2 %3 %4\% %5 %6\%\n")
            .arg("" /* name */, 16)
            .arg("" /* wins */, 14)
            .arg(sum_black_wins, 4)
            .arg(100.0 * sum_black_wins / m_gamesPlayed, 5, 'g', 4)
            .arg(sum_white_wins, 7)
            .arg(100.0 * sum_white_wins / m_gamesPlayed, 5, 'g', 4);
}

void Validation::quitThreads() {
    for(int gpu = 0; gpu < m_gpus * m_games; ++gpu) {
        m_gamesThreads[gpu].quit();
    }
}
