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

#include "Worker.h"
#include "Game.h"
#include <QTextStream>
#include <QLockFile>
#include <QUuid>
#include <chrono>


Worker::Worker(int index, const QString& gpuIndex, Management *parent) :
    m_index(index),
    m_state(),
    m_gpu(""),
    m_job(nullptr),
    m_boss(parent)
{
    if (!gpuIndex.isEmpty()) {
        m_gpu = " --gpu=" + gpuIndex + " ";
    }
}

void Worker::doStore() {
    QTextStream(stdout) << "Storing current game ..." << endl;
    m_job->store();
    m_state.store(STORING);
}

void Worker::order(Order o)
{
    if (!o.isValid()) {
        if (m_job != nullptr) {
            m_job->finish();
        }
        return;
    }
    if (m_todo.type() != o.type() || m_job == nullptr) {
        createJob(o.type());
    }
    m_todo = o;
    m_job->init(m_todo);
}


void Worker::createJob(int type) {
    if (m_job != nullptr) {
        delete m_job;
    }
    switch (type) {
    case Order::Production:
    case Order::RestoreSelfPlayed:
        m_job = new ProductionJob(m_gpu, m_boss);
        break;
    case Order::Validation:
    case Order::RestoreMatch:
        m_job = new ValidationJob(m_gpu, m_boss);
        break;
    case Order::Wait:
        m_job = new WaitJob(m_gpu, m_boss);
        break;
    }
}

void Worker::run() {
     Result res;
     do {
        auto start = std::chrono::high_resolution_clock::now();
        res = m_job->execute();
        auto end = std::chrono::high_resolution_clock::now();
        auto gameDuration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        if (m_state != STORING) {
            emit resultReady(m_todo, res, m_index, gameDuration);
        }
    } while (m_state == RUNNING);
    if (m_state == STORING) {
        m_todo.add("moves", res.parameters()["moves"]);
        m_todo.add("sgf", res.parameters()["sgf"]);
        if (res.type() == Result::StoreMatch) {
            m_todo.type(Order::RestoreMatch);
        } else {
            m_todo.type(Order::RestoreSelfPlayed);
        }
        QString unique = QUuid::createUuid().toRfc4122().toHex();
        QLockFile fi("storefile" + unique + ".bin.lock");
        fi.lock();
        m_todo.save("storefile" + unique + ".bin");
        fi.unlock();
    }
    QTextStream(stdout) << "Program ends: quitting current worker." << endl;
}
