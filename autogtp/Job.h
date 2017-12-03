#ifndef JOB_H
#define JOB_H

#include "Result.h"
#include <QObject>
#include <QAtomicInt>

class Job : public QObject {
    Q_OBJECT
public:
    enum {
        RUNNING = 0,
        FINISHING
    };
    enum {
        Production = 0,
        Validation
    };
    Job();
    ~Job() = default;
    virtual Result execute() = 0;
    virtual void init(const QStringList &l) = 0;
    void finish() { m_state.store(FINISHING); }

protected:
    QAtomicInt m_state;
    QString m_option;
};


class ProdutionJob : public Job {
    Q_OBJECT
public:
    ProdutionJob();
    ~ProdutionJob() = default;
    void init(const QStringList &l);
    Result execute();
private:
    QString m_network;
};

class ValidationJob : public Job {
    Q_OBJECT
public:
    ValidationJob();
    ~ValidationJob() = default;
    void init(const QStringList &l);
    Result execute();
private:
    QString m_firstNet;
    QString m_secondNet;
};

#endif // JOB_H
