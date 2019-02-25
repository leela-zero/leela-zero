#ifndef VALIDATION_H
#define VALIDATION_H
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

#include <QTextStream>
#include <QString>
#include <QThread>
#include <QVector>
#include <QAtomicInt>
#include <QMutex>
#include "SPRT.h"
#include "../autogtp/Game.h"
#include "Results.h"

class ValidationWorker : public QThread {
    Q_OBJECT
public:

    enum {
        RUNNING = 0,
        FINISHING
    };
    ValidationWorker() = default;
    ValidationWorker(const ValidationWorker& w) : QThread(w.parent()) {}
    ~ValidationWorker() = default;
    void init(const QString& gpuIndex,
              const QVector<Engine>& engines,
              const QString& keep,
              int expected);
    void run() override;
    void doFinish() { m_state.store(FINISHING); }

signals:
    void resultReady(Sprt::GameResult r, int net_one_color);
private:
    QVector<Engine> m_engines;
    int m_expected;
    QString m_keepPath;
    QAtomicInt m_state;
};

class Validation : public QObject {
    Q_OBJECT

public:
    Validation(const int gpus, const int games,
               const QStringList& gpusList,
               QVector<Engine>& engines,
               const QString& keep,
               QMutex* mutex,
               const float& h0,
               const float& h1);
    ~Validation() = default;
    void startGames();
    void wait();
    void loadSprt();
signals:
    void sendQuit();
public slots:
    void getResult(Sprt::GameResult result, int net_one_color);
    void storeSprt();
private:
    QMutex* m_mainMutex;
    QMutex m_syncMutex;
    Sprt m_statistic;
    Results m_results;
    QVector<ValidationWorker> m_gamesThreads;
    int m_games;
    int m_gpus;
    QStringList m_gpusList;
    QVector<Engine>& m_engines;
    QString m_keepPath;
    void quitThreads();
    void saveSprt();
    void printSprtStatus(const Sprt::Status& status);
};

#endif
