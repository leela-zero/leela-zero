#include "Distributedval.h"

Distributedval::Distributedval(const QString& keep):
    QThread(),
  m_syncMutex(),
  m_game(),
  m_firstNet(""),
  m_secondNet(""),
  m_keepPath(keep),
  m_state(ValidationWorker::RUNNING),
  m_color(Game::BLACK)
{
    connect(&m_game, &ValidationWorker::resultReady,
            this, &Distributedval::getResult,
            Qt::DirectConnection);
}

void Distributedval::run() {
    do {
        if(!hasToWork())
        {
            QThread::sleep(60);
        } else {
            getNetworks();
            int m_color = getColor();
            if(m_color == Game::BLACK) {
                m_game.init("", m_firstNet, m_secondNet, m_keepPath, m_color);
            } else {
                m_game.init("", m_secondNet, m_firstNet, m_keepPath, m_color);
            }
            m_game.start();
            m_game.wait();
        }
    } while(m_state.load() == ValidationWorker::RUNNING)
}

void Distributedval::resultReady(Sprt::GameResult r) {
    if(result == Sprt::NoResult) {
        QTextStream(stdout) << "Engine Error." << endl;
        return;
    }
    m_syncMutex.lock();
    sendGame(r);
    m_game.doFinish();
    m_syncMutex.unlock();
}

bool Distributedval::hasToWork();
bool Distributedval::getNetworks();
int Distributedval::getColor();
void Distributedval::sendGame(Sprt::GameResult r);
