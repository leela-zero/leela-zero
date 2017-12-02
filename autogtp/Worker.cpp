#include "Worker.h"
#include "Game.h"
#include <QTextStream>
#include <chrono>

Worker::Worker(int index, int job) :
    m_mutex(),
    m_state(),
    m_option(" -g -q -d -r 0 -w "),
    m_index(index),
    m_job(job)
{
}


ProductionWorker::ProductionWorker(int index) :
    Worker(index, Worker::PRODUCTION),
    m_network()
{
}

ValidationWorker::ValidationWorker(int index) :
    Worker(index, Worker::VALIDATION),
    m_firstNet(),
    m_secondNet(),
    m_expected(Game::BLACK),
    m_keepPath()
{
}

void Worker::doInit(const QString &gpuIndex) {
    if (!gpuIndex.isEmpty()) {
        m_option.prepend(" --gpu=" + gpuIndex + " ");
    }
}

void ProductionWorker::run() {
     do {
         auto start = std::chrono::high_resolution_clock::now();
         m_mutex.lock();
         Game game(m_network, m_option);
         m_mutex.unlock();
         if(!game.gameStart(min_leelaz_version)) {
             return;
         }
         do {
             game.move();
             if(!game.waitForMove()) {
                 return;
             }
             game.readMove();
         } while (game.nextMove() && m_state == RUNNING);
         switch(m_state) {
         case RUNNING:
         {
             QTextStream(stdout) << "Game has ended." << endl;
             if (game.getScore()) {
                 game.writeSgf();
                 game.dumpTraining();
             }
             auto end = std::chrono::high_resolution_clock::now();
             auto gameDuration =
                std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
             emit resultReady(game.getFile(), gameDuration, m_index);
             QTextStream(stdout) << "Stopping engine." << endl;
             game.gameQuit();
             break;
         }
         case NET_CHANGE:
         {
             QTextStream(stdout) << "Best Network has change: restarting." << endl;
             m_state = RUNNING;
             QTextStream(stdout) << "Stopping engine." << endl;
             game.gameQuit();
             break;
         }
         case FINISHING:
         {
             QTextStream(stdout) << "Program ends: exiting." << endl;
             QTextStream(stdout) << "Stopping engine." << endl;
             game.gameQuit();
             break;
         }
        }
    } while (m_state != FINISHING);
}

void ProductionWorker::init(const QString& gpuIndex, const QString& net) {
    Worker::doInit(gpuIndex);
    m_option.prepend(" -n -m 30 ");
    m_network = net;
    m_state.store(Worker::RUNNING);
}

void ValidationWorker::run() {
     do {
        Game first(m_firstNet,  m_option);
        if(!first.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, m_index);
            return;
        }
        Game second(m_secondNet, m_option);
        if(!second.gameStart(min_leelaz_version)) {
            emit resultReady(Sprt::NoResult, m_index);
            return;
        }
        QString wmove = "play white ";
        QString bmove = "play black ";
        do {
            first.move();
            if(!first.waitForMove()) {
                emit resultReady(Sprt::NoResult, m_index);
                return;
            }
            first.readMove();
            second.setMove(bmove + first.getMove());
            second.move();
            if(!second.waitForMove()) {
                emit resultReady(Sprt::NoResult, m_index);
                return;
            }
            second.readMove();
            first.setMove(wmove + second.getMove());
            second.nextMove();
        } while (first.nextMove() && m_state.load() == RUNNING);

        if(m_state.load() == RUNNING) {
            QTextStream(stdout) << "Game has ended." << endl;
            int result = 0;
            if (first.getScore()) {
                result = first.getWinner();
            }
            if(!m_keepPath.isEmpty())
            {
                first.writeSgf();
                QString prefix = m_keepPath + '/';
                if(m_expected == Game::BLACK) {
                    prefix.append("black_");
                } else {
                    prefix.append("white_");
                }
                QFile(first.getFile() + ".sgf").rename(prefix + first.getFile() + ".sgf");
                }
            QTextStream(stdout) << "Stopping engine." << endl;
            first.gameQuit();
            second.gameQuit();

            // Game is finished, send the result
            if (result == m_expected) {
                emit resultReady(Sprt::Win, m_index);
            } else {
                emit resultReady(Sprt::Loss, m_index);
            }
            // Change color and play again
            QString net;
            net = m_secondNet;
            m_secondNet = m_firstNet;
            m_firstNet = net;
            if(m_expected == Game::BLACK) {
                m_expected = Game::WHITE;
            } else {
                m_expected = Game::BLACK;
            }
        }
    } while (m_state.load() != FINISHING);
}

void ValidationWorker::init(const QString& gpuIndex,
                            const QString& firstNet,
                            const QString& secondNet,
                            const QString& keep,
                            const int expected) {
    m_firstNet = firstNet;
    m_secondNet = secondNet;
    m_expected = expected;
    m_keepPath = keep;
    m_state.store(Worker::RUNNING);
}
