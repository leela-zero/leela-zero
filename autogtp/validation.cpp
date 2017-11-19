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

#include "validation.h"
#include "Game.h"


 void WorkerThread::run(){
     do {
        Game first(m_firstNet,  " -g -q  -r 0 -w ");
        if(!first.gameStart()) {
            emit resultReady(Sprt::NoResult);
            return;
        }
        Game second(m_secondNet, " -g -q  -r 0 -w ");
        if(!second.gameStart()) {
            emit resultReady(Sprt::NoResult);
            return;
        }
        QString wmove = "play white ";
        QString bmove = "play black ";
        do {
            first.move();
            if(!first.waitForMove()) {
                emit resultReady(Sprt::NoResult);
                return;
            }
            first.readMove();
            second.setMove(bmove + first.getMove());
            second.move();
            if(!second.waitForMove()) {
                emit resultReady(Sprt::NoResult);
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
            first.writeSgf();
            second.writeSgf();
            first.dumpTraining();
        }
        QTextStream(stdout) << "Stopping engine." << endl;
        first.gameQuit();
        second.gameQuit();

        //game is finished send the result
        if(result == m_expected)
            emit resultReady(Sprt::Win);
        else
            emit resultReady(Sprt::Loss);
        //change color and play again
        QString net;
        net = m_secondNet;
        m_secondNet = m_firstNet;
        m_firstNet = net;
        if(m_expected == Game::BLACK) {
            m_expected = Game::WHITE;
        }
        else {
            m_expected = Game::BLACK;
        }
    } while (1);
}

void WorkerThread::init(const int &gpu, const int &game, const QString &gpuIndex,
                        const QString &firstNet, const QString &secondNet, const int &expected) {
    m_gpu = gpu;
    m_game = game;
    m_option = " -g -q  -r 0 -w ";
    if(!gpuIndex.isEmpty())
        m_option.append(" -gpu=" + gpuIndex);
    m_firstNet = firstNet;
    m_secondNet = secondNet;
    m_expected = expected;
 }


Validation::Validation(const int &gpus, const int &games, const QStringList &gpuslist,
                       const QString &firstNet, const QString &secondNet, QMutex *mutex) :
    m_mainMutex(mutex),
    m_syncMutex(),
    m_gamesThreads(gpus*games),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_gamesPlayed(0),
    m_firstNet(firstNet),
    m_secondNet(secondNet)
{
    m_statistic.initialize(25.0,35.0,0.05,0.05);
    m_statistic.addGameResult(Sprt::Win);
    m_statistic.addGameResult(Sprt::Loss);
    m_statistic.addGameResult(Sprt::Draw);
}

Validation::~Validation() {
}

void Validation::startGames() { 
    m_mainMutex->lock();
    QString n1, n2;
    int expected;

    for(int gpu = 0; gpu < m_gpus; ++gpu) {
        for(int game = 0; game < m_games; ++game) {
            connect(&m_gamesThreads[gpu * m_games + game], &WorkerThread::resultReady, this, &Validation::getResult, Qt::DirectConnection);
            if(game % 2) {
                n1 = m_firstNet;
                n2 = m_secondNet;
                expected = Game::BLACK;
            }
            else {
                n1 = m_secondNet;
                n2 = m_firstNet;
                expected = Game::WHITE;
            }
            if(m_gpusList.isEmpty()) {
                m_gamesThreads[gpu * m_games + game].init(gpu, game, "", n1, n2, expected);
            }
            else {
                m_gamesThreads[gpu * m_games + game].init(gpu, game, m_gpusList.at(gpu), n1, n2, expected);
            }
            m_gamesThreads[gpu * m_games + game].start();
        }
    }
}

void Validation::getResult(Sprt::GameResult result) {
    if(result == Sprt::NoResult) {
        QTextStream(stdout) << "Engine Error." << endl;
        return;
    }
    m_syncMutex.lock();
    m_gamesPlayed++;
    m_statistic.addGameResult(result);
    Sprt::Status status = m_statistic.status();
    if(status.result != Sprt::Continue) {
        quitThreads();
        QTextStream(stdout) << "The first net is ";
        QTextStream(stdout) <<  ((status.result ==  Sprt::AcceptH0) ? "wrost " : "better ");
        QTextStream(stdout) << "then the second" << endl;
        m_mainMutex->unlock();
    }
    else {
        QTextStream(stdout) << m_gamesPlayed << " games played." << endl;
        QTextStream(stdout) << "Status: " << status.result << " Log-Likelyhood Ratio ";
        QTextStream(stdout) << status.llr <<  " Lower Bound " << status.lBound ;
        QTextStream(stdout) << " Upper Bound " << status.uBound << endl;
    }
    m_syncMutex.unlock();
}

void Validation::quitThreads() {
    for(int gpu = 0; gpu < m_gpus * m_games; ++gpu) {
        m_gamesThreads[gpu].quit();
    }
}

