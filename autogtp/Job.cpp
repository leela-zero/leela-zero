/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Marco Calignano

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
    m_gpu(gpu),
    m_boss(parent)
{
}

void Job::init(const Order &o) {
    QStringList version_list = o.parameters()["leelazVer"].split(".");
    if (version_list.size() < 2) {
        QTextStream(stdout)
                << "Unexpected Leela Zero version: " << o.parameters()["leelazVer"] << endl;
        exit(EXIT_FAILURE);
    }
    if (version_list.size() < 3) {
        version_list.append("0");
    }
    std::get<0>(m_leelazMinVersion) = version_list[0].toInt();
    std::get<1>(m_leelazMinVersion) = version_list[1].toInt();
    std::get<2>(m_leelazMinVersion) = version_list[2].toInt();
}

ProductionJob::ProductionJob(QString gpu, Management *parent) :
    Job(gpu, parent),
    m_engine(Engine(QString(), QString()))
{
}

ValidationJob::ValidationJob(QString gpu, Management *parent) :
    Job(gpu, parent),
    m_engineFirst(Engine(QString(), QString())),
    m_engineSecond(Engine(QString(), QString()))
{
}

WaitJob::WaitJob(QString gpu, Management *parent) :
    Job(gpu, parent)
{
}

Result ProductionJob::execute(){
    Result res(Result::Error);
    Game game(m_engine);
    if (!game.gameStart(m_leelazMinVersion, m_sgf, m_moves)) {
        return res;
    }
    if (!m_sgf.isEmpty()) {
        QFile::remove(m_sgf + ".sgf");
        if (m_restore) {
            game.loadTraining(m_sgf);
            QFile::remove(m_sgf + ".train");
        }
    }
    do {
        game.move();
        if (!game.waitForMove()) {
            return res;
        }
        game.readMove();
        m_boss->incMoves();
    } while (game.nextMove() && m_state.load() == RUNNING);
    switch (m_state.load()) {
    case RUNNING:
        QTextStream(stdout) << "Game has ended." << endl;
        if (game.getScore()) {
            game.writeSgf();
            game.fixSgf(m_engine, false, true);
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
        game.writeSgf();
        game.saveTraining();
        res.type(Result::StoreSelfPlayed);
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
    m_engine.m_network = "networks/" + o.parameters()["network"] + ".gz";
    m_engine.m_options = " " + o.parameters()["options"] + m_gpu + " -g -q -w ";
    if (o.parameters().contains("gtpCommands")) {
        m_engine.m_commands = o.parameters()["gtpCommands"].split(",");
    }
    m_debug = o.parameters()["debug"] == "true";
    m_sgf = o.parameters()["sgf"];
    m_moves = o.parameters()["moves"].toInt();
    m_restore = o.type() == Order::RestoreSelfPlayed;
}

Result ValidationJob::execute(){
    Result res(Result::Error);
    Game first(m_engineFirst);
    if (!first.gameStart(m_leelazMinVersion, m_sgf, m_moves)) {
        return res;
    }
    Game second(m_engineSecond);
    if (!second.gameStart(m_leelazMinVersion, m_sgf, m_moves)) {
        return res;
    }
    if (!m_sgf.isEmpty()) {
        QFile::remove(m_sgf + ".sgf");
    }

    const QString stringWhite = "white";
    const QString stringBlack = "black";
    //Start with the side to move set to the opposite of the expected way around
    //because the game playing loop swaps the sides at the start of each iteration.
    //This avoids having to test which side is to move on every iteration of the loop.
    auto gameToMove = &second;
    auto colorToMove = &stringWhite;
    auto gameOpponent = &first;
    auto colorOpponent = &stringBlack;
    if (first.getToMove() == Game::WHITE) {
        std::swap(gameToMove, gameOpponent);
        std::swap(colorToMove, colorOpponent);
    }
    do {
        std::swap(gameToMove, gameOpponent);
        std::swap(colorToMove, colorOpponent);
        gameToMove->move();
        if (!gameToMove->waitForMove()) {
            return res;
        }
        gameToMove->readMove();
        m_boss->incMoves();
        gameOpponent->setMove("play " + *colorToMove + " " + gameToMove->getMove());
    } while (gameToMove->nextMove() && m_state.load() == RUNNING);

    switch (m_state.load()) {
    case RUNNING:
        QTextStream(stdout) << "Game has ended." << endl;
        if (first.getScore()) {
            res.add("score", first.getResult());
            res.add("winner", first.getWinnerName());
            first.writeSgf();
            first.fixSgf(m_engineSecond,
                (res.parameters()["score"] == "B+Resign"),
                false);
            res.add("file", first.getFile());
        }
        res.type(Result::Win);
        res.add("moves", QString::number(first.getMovesCount()));
        break;
    case STORING:
        first.writeSgf();
        res.type(Result::StoreMatch);
        res.add("sgf", first.getFile());
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
    m_engineFirst.m_network = "networks/" + o.parameters()["firstNet"] + ".gz";
    m_engineFirst.m_options = " " + o.parameters()["options"] + m_gpu + " -g -q -w ";
    if (o.parameters().contains("gtpCommands")) {
        m_engineFirst.m_commands = o.parameters()["gtpCommands"].split(",");
    }
    m_engineSecond.m_network = "networks/" + o.parameters()["secondNet"] + ".gz";
    m_engineSecond.m_options = " " + o.parameters()["optionsSecond"] + m_gpu + " -g -q -w ";
    if (o.parameters().contains("gtpCommandsSecond")) {
        m_engineSecond.m_commands = o.parameters()["gtpCommandsSecond"].split(",");
    }
    m_sgf = o.parameters()["sgf"];
    m_moves = o.parameters()["moves"].toInt();
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
