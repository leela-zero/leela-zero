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

#include "Worker.h"
#include "Game.h"
#include <QTextStream>
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
     m_job->init(m_todo.parameters());
}


void Worker::createJob(int type) {
    if (m_job != nullptr) {
        delete m_job;
    }
    switch(type) {
    case Order::Production:
        m_job = new ProductionJob(m_gpu, m_boss);
        break;
    case Order::Validation:
        m_job = new ValidationJob(m_gpu, m_boss);
        break;
    }
}

void Worker::run() {
     do {
        auto start = std::chrono::high_resolution_clock::now();

        Result res = m_job->execute();

        auto end = std::chrono::high_resolution_clock::now();
        auto gameDuration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        emit resultReady(m_todo, res, m_index, gameDuration);
    } while (m_state != FINISHING);
    QTextStream(stdout) << "Program ends: exiting." << endl;
}
