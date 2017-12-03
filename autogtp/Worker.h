#ifndef WORKER_H
#define WORKER_H

#include "Job.h"
#include "Order.h"
#include <QThread>
#include <QMutex>

class Worker : public QThread {
   Q_OBJECT
public:
    enum {
        RUNNING = 0,
        FINISHING
    };
    Worker(int index, const QString& gpuIndex,const QString& keep);
    ~Worker() = default;
    void order(Order o);
    void doFinish() { m_job->finish(); m_state.store(FINISHING); }
    void run() override;
signals:
    void resultReady(Order o, Result r, int index, int duration);
private:
    int m_index;
    QAtomicInt m_state;
    QString m_keepPath;
    QString m_option;
    Order m_todo;
    Job *m_job;
    void createJob(int type);
};

#endif // WORKER_H
