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
#include "Management.h"
#include <QTextStream>
#include <chrono>

#include <QThread>

Job::Job(QString gpu, Management *parent) :
  m_state(RUNNING),
  m_option(""),
  m_gpu(gpu),
  m_boss(parent)
{
}

void Job::init(const QMap<QString,QString> &l) {
   m_option = l["options"] + m_gpu + " -g -q -w ";
   QStringList version_list = l["leelazVer"].split(".");
   if (version_list.size() < 2) {
        QTextStream(stdout)
             << "Unexpected Leela Zero version: " << l[0] << endl;
        exit(EXIT_FAILURE);
    }
    std::get<0>(m_leelazMinVersion) = version_list[0].toInt();
    std::get<1>(m_leelazMinVersion) = version_list[1].toInt();
}

ProductionJob::ProductionJob(QString gpu, Management *parent) :
Job(gpu, parent)
{
}

ValidationJob::ValidationJob(QString gpu, Management *parent) :
Job(gpu, parent)
{
}

Result ProductionJob::execute(){
    Result res(Result::Error);
    Game game(m_network, m_option);
    if (!game.gameStart(m_leelazMinVersion)) {
        return res;
    }
    do {
        game.move();
        if (!game.waitForMove()) {
            return res;
        }
        game.readMove();
        m_boss->incMoves();
    } while (game.nextMove() && m_state.load() == RUNNING);
    if (m_state.load() == RUNNING) {
        QTextStream(stdout) << "Game has ended." << endl;
        if (game.getScore()) {
            game.writeSgf();
            game.fixSgf(m_network, false);
            game.dumpTraining();
            if (m_debug) {
                game.dumpDebug();
            }
        }
        res.type(Result::File);
        res.add("file", game.getFile());
        res.add("winner", game.getWinnerName());
        res.add("moves", QString::number(game.getMovesCount()));
    } else {
        QTextStream(stdout) << "Program ends: exiting." << endl;
    }
    QTextStream(stdout) << "Stopping engine." << endl;
    game.gameQuit();
    return res;
}

void ProductionJob::init(const QMap<QString,QString> &l) {
    Job::init(l);
    m_network = l["network"];
    m_debug = l["debug"] == "true";
}


Result ValidationJob::execute(){
   Result res(Result::Error);
   Game first(m_firstNet,  m_option);
   if (!first.gameStart(m_leelazMinVersion)) {
       return res;
   }
   Game second(m_secondNet, m_option);
   if (!second.gameStart(m_leelazMinVersion)) {
       return res;
   }
   QString wmove = "play white ";
   QString bmove = "play black ";
   do {
       first.move();
       if (!first.waitForMove()) {
           return res;
       }
       first.readMove();
       m_boss->incMoves();
       if (first.checkGameEnd()) {
           break;
       }
       second.setMove(bmove + first.getMove());
       second.move();
       if (!second.waitForMove()) {
           return res;
       }
       second.readMove();
       m_boss->incMoves();
       first.setMove(wmove + second.getMove());
       second.nextMove();
   } while (first.nextMove() && m_state.load() == RUNNING);

   if (m_state.load() == RUNNING) {
       res.add("moves", QString::number(first.getMovesCount()));
       QTextStream(stdout) << "Game has ended." << endl;
       if (first.getScore()) {
           res.add("score", first.getResult());
           res.add("winner", first.getWinnerName());
           first.writeSgf();
           first.fixSgf(m_secondNet, (res.parameters()["score"] == "B+Resign"));
           res.add("file", first.getFile());
       }
       // Game is finished, send the result
       res.type(Result::Win);
   }
   QTextStream(stdout) << "Stopping engine." << endl;
   first.gameQuit();
   second.gameQuit();
   return res;
}


void ValidationJob::init(const QMap<QString,QString> &l) {
    Job::init(l);
    m_firstNet = l["firstNet"];
    m_secondNet = l["secondNet"];
}





