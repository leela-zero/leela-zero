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

#include <QAtomicInt>
#include <QMutex>
#include <QString>
#include <QTextStream>
#include <QThread>
#include <QVector>
#include <chrono>
#include <stdexcept>

class ProductionWorker : public QThread {
    Q_OBJECT
public:
    enum {
        RUNNING = 0,
        NET_CHANGE,
        FINISHING
    };
    ProductionWorker() = default;
    ProductionWorker(const ProductionWorker& w) : QThread(w.parent()) {}
    ~ProductionWorker() = default;
    void init(const QString& gpuIndex, const QString& net, QAtomicInt* movesMade);
    void newNetwork(const QString& net) {
        QMutexLocker locker(&m_mutex);
        m_state = NET_CHANGE;
        m_network = net;
    }
    void run() override;
    
signals:
    void resultReady(const QString& file, float duration);
private:
    QAtomicInt* m_movesMade;
    QString m_network;
    QString m_option;
    QMutex m_mutex;
    int m_state;
};

class Production : public QObject {
    Q_OBJECT

public:
    Production(const int gpus,
               const int games,
               const QStringList& gpuslist,
               const int ver,
               const QString& keep,
               const QString& debug,
               QMutex* mutex);
    ~Production() = default;
    void startGames();

public slots:
    void getResult(const QString& file, float duration);

private:

    struct NetworkException: public std::runtime_error
    {
        NetworkException(std::string const& message)
            : std::runtime_error("NetworkException: " + message)
        {}
    };


    QMutex* m_mainMutex;
    QMutex m_syncMutex;
    QVector<ProductionWorker> m_gamesThreads;
    int m_games;
    int m_gpus;
    QStringList m_gpusList;
    int m_gamesPlayed;
    QAtomicInt m_movesMade;
    QString m_network;
    QString m_keepPath;
    QString m_debugPath;
    int m_version;
    std::chrono::high_resolution_clock::time_point m_start;
    bool fetchBestNetworkHash();
    void fetchBestNetwork();
    void uploadData(const QString& file);
    void printTimingInfo(float duration);
    bool updateNetwork();
    bool networkExists();
};

#endif
