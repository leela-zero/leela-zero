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
#ifndef MANAGEMENT_H
#define MANAGEMENT_H

#include <QAtomicInt>
#include <QMutex>
#include <QString>
#include <QTextStream>
#include <QThread>
#include <QVector>
#include <chrono>
#include <stdexcept>
#include "Worker.h"

class Management : public QObject {
    Q_OBJECT
public:
    Management(const int gpus,
               const int games,
               const QStringList& gpuslist,
               const int ver,
               const QString& keep,
               const QString& debug,
               QMutex* mutex);
    ~Management() = default;
    void giveAssignments();
    void incMoves() { m_movesMade++; }
public slots:
    void getResult(Order ord, Result res, int index, int duration);

private:

    struct NetworkException: public std::runtime_error
    {
        NetworkException(std::string const& message)
            : std::runtime_error("NetworkException: " + message)
        {}
    };
    QMutex* m_mainMutex;
    QMutex m_syncMutex;
    QVector<Worker*> m_gamesThreads;
    int m_games;
    int m_gpus;
    QStringList m_gpusList;
    int m_selfGames;
    int m_matchGames;
    int m_gamesPlayed;
    QAtomicInt m_movesMade;
    QString m_keepPath;
    QString m_debugPath;
    int m_version;
    std::chrono::high_resolution_clock::time_point m_start;
    Order m_fallBack;
    Order getWorkInternal();
    Order getWork();
    QString getOption(const QJsonObject &ob, const QString &key, const QString &opt, const QString &defValue);
    QString getBoolOption(const QJsonObject &ob, const QString &key, const QString &opt, bool defValue);
    void sendAllGames();
    bool networkExists(const QString &name);
    void fetchNetwork(const QString &name);
    void printTimingInfo(float duration);
    void gzipFile(const QString &fileName);
    bool sendCurl(const QStringList &lines);
    void saveCurlCmdLine(const QStringList &prog_cmdline, const QString &name);
    void archiveFiles(const QString &fileName);
    void cleanupFiles(const QString &fileName);
    void uploadData(const QMap<QString,QString> &r, const QMap<QString,QString> &l);
    void uploadResult(const QMap<QString, QString> &r, const QMap<QString, QString> &l);
};

#endif
