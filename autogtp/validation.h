#ifndef VALIDATION_H 
#define VALIDATION_H

#include <QTextStream>
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
