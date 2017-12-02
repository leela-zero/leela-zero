#ifndef WORKER_H
#define WORKER_H

#include <QThread>
#include <QMutex>

class Worker : public QThread {
   Q_OBJECT
public:
    enum {
        RUNNING = 0,
        NET_CHANGE,
        FINISHING
    };
    enum {
        PRODUCTION = 0,
        VALIDATION
    };
    Worker(int index, int job);
    ~Worker() = default;
private:
    QMutex m_mutex;
    QAtomicInt m_state;
    QString m_option;
    int m_index;
    int m_job;
    void doFinish() { m_state.store(FINISHING); }
    void doInit(const QString& gpuIndex);
    int myJob() { return m_job; }
};


class ProductionWorker : public Worker {
    Q_OBJECT
public:
    ProductionWorker(int index);
    ~ProductionWorker() = default;
    void init(const QString& gpuIndex, const QString& net);
    void newNetwork(const QString& net) { QMutexLocker locker(&m_mutex); m_state.store(NET_CHANGE); m_network = net; }
    void run() override;

signals:
    void resultReady(const QString& file, float duration, int index);
private:
    QString m_network;
};

class ValidationWorker : public Worker {
    Q_OBJECT
public:
    enum {
        RUNNING = 0,
        FINISHING
    };
    ValidationWorker(int index);
    ~ValidationWorker() = default;
    void init(const QString& gpuIndex,
              const QString& firstNet,
              const QString& secondNet,
              const QString& keep,
              int expected);
    void run() override;

signals:
    void resultReady(Sprt::GameResult r, int index);
private:
    QString m_firstNet;
    QString m_secondNet;
    int m_expected;
    QString m_keepPath;
};


#endif // WORKER_H
