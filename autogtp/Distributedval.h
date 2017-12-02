#ifndef DISTRIBUTEDVAL_H
#define DISTRIBUTEDVAL_H

#include "Validation.h"

class Distributedval : public QThread {
    Q_OBJECT
public:
    Distributedval(const QString& keep);
    ~Distributedval() = default;
    void run();

signals:
    void resultReady(Sprt::GameResult r);

private:
    QMutex m_syncMutex;
    ValidationWorker m_game;
    QString m_firstNet;
    QString m_secondNet;
    QString m_keepPath;
    QAtomicInt m_state;
    int m_color;
    bool hasToWork();
    bool getNetworks();
    int getColor();
    sendGame(Sprt::GameResult r);
};

#endif // DISTRIBUTEDVAL_H
