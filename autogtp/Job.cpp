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

#include "Job.h"
#include "Game.h"
#include <QTextStream>
#include <chrono>

#include <QThread>

Job::Job(QString gpu) :
  m_state(RUNNING),
  m_option(""),
  m_gpu(gpu)
{
}

ProdutionJob::ProdutionJob(QString gpu) :
Job(gpu)
{
}

ValidationJob::ValidationJob(QString gpu) :
Job(gpu)
{
}

Result ProdutionJob::execute(){
    Result res(Result::Error);
    Game game(m_network, m_option);
    if (!game.gameStart(min_leelaz_version)) {
        return res;
    }
    do {
        game.move();
        if(!game.waitForMove()) {
            return res;
        }
        game.readMove();
    } while (game.nextMove() && m_state.load() == RUNNING);
    if (m_state.load() == RUNNING) {
        QTextStream(stdout) << "Game has ended." << endl;
        if (game.getScore()) {
            game.writeSgf();
            game.dumpTraining();
        }
        res.type(Result::File);
        res.addList(game.getFile());
        res.addList(QString::number(game.getMovesCount()));
    } else {
        QTextStream(stdout) << "Program ends: exiting." << endl;
    }
    QTextStream(stdout) << "Stopping engine." << endl;
    game.gameQuit();
    return res;
}

void ProdutionJob::init(const QStringList &l) {
    Job::init(l);
    m_network = l[2];
}


Result ValidationJob::execute(){
   Result res(Result::Error);
   Game first(m_firstNet,  m_option);
   if (!first.gameStart(min_leelaz_version)) {
       return res;
   }
   Game second(m_secondNet, m_option);
   if (!second.gameStart(min_leelaz_version)) {
       return res;
   }
   QString wmove = "play white ";
   QString bmove = "play black ";
   do {
       first.move();
       if(!first.waitForMove()) {
           return res;
       }
       first.readMove();
       second.setMove(bmove + first.getMove());
       second.move();
       if(!second.waitForMove()) {
           return res;
       }
       second.readMove();
       first.setMove(wmove + second.getMove());
       second.nextMove();
   } while (first.nextMove() && m_state.load() == RUNNING);

   if (m_state.load() == RUNNING) {
       QTextStream(stdout) << "Game has ended." << endl;
       res.addList(QString::number(first.getMovesCount()));      //[0]
       if (first.getScore()) {
           res.addList(first.getResult());                       //[1]
           res.addList(first.getWinnerName());                   //[2]
           first.writeSgf();
           res.addList(first.getFile());                         //[3]
           res.addList(QString::number(first.getMovesCount()));  //[4]
       }
       // Game is finished, send the result
       res.type(Result::Win);
   }
   QTextStream(stdout) << "Stopping engine." << endl;
   first.gameQuit();
   second.gameQuit();
   return res;
}


void ValidationJob::init(const QStringList &l) {
    Job::init(l);
    m_firstNet = l[2];
    m_secondNet = l[3];
}





