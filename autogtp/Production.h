#ifndef PRODUCTION_H
#define PRODUCTION_H
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

#include <QTextStream>
#include <QString>
#include <QThread>
#include <QVector>
#include <QMutex>
#include <QTextStream>
#include <chrono>

class ProductionWorker : public QThread {
    Q_OBJECT
public:
    ProductionWorker() = default;
    ProductionWorker(const ProductionWorker& w) : QThread(w.parent()) {}
    ~ProductionWorker() = default;
    void init(const QString& gpuIndex, const QString& net);
    void newNetwork(const QString& net) { m_network = net; }
    void run() override;

signals:
    void resultReady(const QString& file, float duration);
private:
    QString m_network;
    QString m_option;
};

class Production : public QObject {
    Q_OBJECT

public:
    Production(const int gpus,
               const int games,
               const QStringList& gpuslist,
               const int ver,
               const QString& keep,
               QMutex* mutex);
    ~Production() = default;
    void startGames();

public slots:
    void getResult(const QString& file, float duration);

private:
    QMutex* m_mainMutex;
    QMutex m_syncMutex;
    QVector<ProductionWorker> m_gamesThreads;
    int m_games;
    int m_gpus;
    QStringList m_gpusList;
    int m_gamesPlayed;
    QString m_network;
    QString m_keepPath;
    int m_version;
    std::chrono::high_resolution_clock::time_point m_start;
    bool fetchBestNetworkHash();
    void fetchBestNetwork();
    void uploadData(const QString& file);
    void printTimingInfo(float duration);
};

#endif
