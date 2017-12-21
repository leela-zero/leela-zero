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

#ifndef WORKER_H
#define WORKER_H

#include "Job.h"
#include "Order.h"

#include <QThread>
#include <QMutex>

class Management;

class Worker : public QThread {
   Q_OBJECT
public:
    enum {
        RUNNING = 0,
        FINISHING
    };
    Worker(int index, const QString& gpuIndex, Management *parent);
    ~Worker() = default;
    void order(Order o);
    void doFinish() { m_job->finish(); m_state.store(FINISHING); }
    void run() override;
signals:
    void resultReady(Order o, Result r, int index, int duration);
private:
    int m_index;
    QAtomicInt m_state;
    QString m_gpu;
    Order m_todo;
    Job *m_job;
    Management *m_boss;
    void createJob(int type);
};

#endif // WORKER_H
