#ifndef VALIDATION_H 
#define VALIDATION_H

#include <QTextStream>
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

#include <QString>
#include <QThread>
#include <QVector>

#include "sprt.h"
class WorkerThread : public QThread
{
    Q_OBJECT
public:
    WorkerThread() {}
    WorkerThread(const WorkerThread& w) : QThread(w.parent()){}
    ~WorkerThread() {}
    void init(const int &gpu, const int &game, const QString &gpuIndex,
              const QString &firstNet, const QString &secondNet, const int &expected);
    void run() override;

signals:
    void resultReady(Sprt::GameResult r);
private:
    int m_gpu;
    int m_game;
    QString m_firstNet;
    QString m_secondNet;
    int m_expected;
    QString m_option;

};


class Validation : public QObject
{
    Q_OBJECT
    
public:
    Validation(const int &gpus, const int &games, const QStringList &gpusList,
               const QString &firstNet, const QString &secondNet, QMutex *mutex);
    ~Validation();
    void startGames();

public slots:
    void getResult(Sprt::GameResult result);

private:
    QMutex *m_mainMutex;
    QMutex m_syncMutex;
    Sprt m_statistic;
    QVector<WorkerThread> m_gamesThreads;
    int m_games;
    int m_gpus;
    QStringList m_gpusList;
    int m_gamesPlayed;
    QString m_firstNet;
    QString m_secondNet; 
    void quitThreads();
};

#endif
