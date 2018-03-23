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

#include <QFile>
#include <QThread>

Job::Job(QString gpu, Management *parent) :
    m_state(RUNNING),
    m_option(""),
  m_gpu(gpu),
  m_boss(parent)
{
}

void Job::init(const Order &o) {
    m_option = " " + o.parameters()["options"] + m_gpu + " -g -q -w ";
    QStringList version_list = o.parameters()["leelazVer"].split(".");
    if (version_list.size() < 2) {
        QTextStream(stdout)
                << "Unexpected Leela Zero version: " << o.parameters()["leelazVer"] << endl;
        exit(EXIT_FAILURE);
    }
   if(version_list.size() < 3) {
       version_list.append("0");
   }
    std::get<0>(m_leelazMinVersion) = version_list[0].toInt();
    std::get<1>(m_leelazMinVersion) = version_list[1].toInt();
    std::get<2>(m_leelazMinVersion) = version_list[2].toInt();

}

ProductionJob::ProductionJob(QString gpu, Management *parent) :
Job(gpu, parent)
{
}

ValidationJob::ValidationJob(QString gpu, Management *parent) :
Job(gpu, parent)
{
}

WaitJob::WaitJob(QString gpu, Management *parent) :
Job(gpu, parent)
{
}

Result ProductionJob::execute(){
    Result res(Result::Error);
    Game game("networks/" + m_network, m_option);
    if (!game.gameStart(m_leelazMinVersion)) {
        return res;
    }
    if(!m_sgf.isEmpty()) {
        game.loadSgf(m_sgf);
        game.loadTraining(m_sgf);
        game.setMovesCount(m_moves);
        QFile::remove(m_sgf + ".sgf");
        QFile::remove(m_sgf + ".train");
    }
    do {
        game.move();
        if (!game.waitForMove()) {
            return res;
        }
        game.readMove();
        m_boss->incMoves();
    } while (game.nextMove() && m_state.load() == RUNNING);
    switch(m_state.load()) {
    case RUNNING:
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
        break;
    case STORING:
        res.type(Result::StoreSelfPlayed);
        game.writeSgf();
        game.saveTraining();
        res.add("sgf", game.getFile());
        res.add("moves", QString::number(game.getMovesCount()));
        break;
    default:
        break;
    }
    game.gameQuit();
    return res;
}

void ProductionJob::init(const Order &o) {
    Job::init(o);
    m_network = o.parameters()["network"];
    m_debug = o.parameters()["debug"] == "true";
    if(o.type() == Order::RestoreSelfPlayed) {
        m_sgf = o.parameters()["sgf"];
        m_moves = o.parameters()["moves"].toInt();
    } else {
        m_sgf = "";
        m_moves = 0;
    }
}

Result ValidationJob::execute(){
    Result res(Result::Error);
    Game first("networks/" + m_firstNet,  m_option);
    if (!first.gameStart(m_leelazMinVersion)) {
        return res;
    }
    if(!m_sgfFirst.isEmpty()) {
        first.loadSgf(m_sgfFirst);
        first.setMovesCount(m_moves);
        QFile::remove(m_sgfFirst + ".sgf");
    }
    Game second("networks/" + m_secondNet, m_option);
    if (!second.gameStart(m_leelazMinVersion)) {
        return res;
    }
    if(!m_sgfSecond.isEmpty()) {
        second.loadSgf(m_sgfSecond);
        second.setMovesCount(m_moves);
        QFile::remove(m_sgfSecond + ".sgf");
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

    switch(m_state.load()) {
    case RUNNING:
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
        break;
    case STORING:
        res.type(Result::StoreMatch);
        first.writeSgf();
        second.writeSgf();
        res.add("sgfFirst", first.getFile());
        res.add("sgfSecond", second.getFile());
        res.add("moves", QString::number(first.getMovesCount()));
        break;
    default:
        break;
    }
    first.gameQuit();
    second.gameQuit();
    return res;
}

void ValidationJob::init(const Order &o) {
    Job::init(o);
    m_firstNet = o.parameters()["firstNet"];
    m_secondNet = o.parameters()["secondNet"];
    if(o.type() == Order::RestoreMatch) {
        m_sgfFirst = o.parameters()["sgfFirst"];
        m_sgfSecond = o.parameters()["sgfSecond"];
        m_moves = o.parameters()["moves"].toInt();
    } else {
        m_sgfFirst = "";
        m_sgfSecond = "";
        m_moves = 0;
    }
}

Result WaitJob::execute(){
    Result res(Result::Waited);
    QThread::sleep(m_minutes * 60);
    return res;
}

void WaitJob::init(const Order &o) {
    Job::init(o);
    m_minutes = o.parameters()["minutes"].toInt();
}


