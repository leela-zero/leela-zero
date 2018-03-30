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
#include <QFileInfo>
#include <QLockFile>
#include <QVector>
#include <chrono>
#include <stdexcept>
#include "Worker.h"

constexpr int AUTOGTP_VERSION = 15;

class Management : public QObject {
    Q_OBJECT
public:
    Management(const int gpus,
               const int games,
               const QStringList& gpuslist,
               const int ver,
               const int maxGame,
               const bool delNetworks,
               const QString& keep,
               const QString& debug);
    ~Management() = default;
    void giveAssignments();
    void incMoves() { m_movesMade++; }
    void wait();
signals:
    void sendQuit();
public slots:
    void getResult(Order ord, Result res, int index, int duration);
    void storeGames();

private:

    struct NetworkException: public std::runtime_error
    {
        NetworkException(std::string const& message)
            : std::runtime_error("NetworkException: " + message)
        {}
    };
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
    int m_storeGames;
    QList<QFileInfo> m_storedFiles;
    Order m_fallBack;
    Order m_lastMatch;
    int m_gamesLeft;
    int m_threadsLeft;
    bool m_delNetworks;
    QLockFile *m_lockFile;

    Order getWorkInternal(bool tuning);
    Order getWork(bool tuning = false);
    Order getWork(const QFileInfo &file);
    QString getOption(const QJsonObject &ob, const QString &key, const QString &opt, const QString &defValue);
    QString getBoolOption(const QJsonObject &ob, const QString &key, const QString &opt, bool defValue);
    QString getOptionsString(const QJsonObject &opt, const QString &rnd);
    void sendAllGames();
    void checkStoredGames();
    QFileInfo getNextStored();
    bool networkExists(const QString &name);
    void fetchNetwork(const QString &net);
    void printTimingInfo(float duration);
    void runTuningProcess(const QString &tuneCmdLine);
    void gzipFile(const QString &fileName);
    bool sendCurl(const QStringList &lines);
    void saveCurlCmdLine(const QStringList &prog_cmdline, const QString &name);
    void archiveFiles(const QString &fileName);
    void cleanupFiles(const QString &fileName);
    void uploadData(const QMap<QString,QString> &r, const QMap<QString,QString> &l);
    void uploadResult(const QMap<QString, QString> &r, const QMap<QString, QString> &l);
};

#endif
