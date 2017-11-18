#include "validation.h"
#include "Game.h"


 void WorkerThread::run(){
        Game first(m_firstNet,  " -g -q  -r 0 -w ");
        if(!first.gameStart()) {
            emit resultReady(Sprt::NoResult, m_gpu, m_game, 0);
            return;
        }
        Game second(m_secondNet, " -g -q  -r 0 -w ");
        if(!second.gameStart()) {
            emit resultReady(Sprt::NoResult, m_gpu, m_game, 0);
            return;
        }
        QString wmove = "play white ";
        QString bmove = "play black ";
        do {
            first.move();
            if(!first.waitForMove()) {
                emit resultReady(Sprt::NoResult, m_gpu, m_game, 0);
                return;
            }
            first.readMove();
            second.setMove(bmove + first.getMove());
            second.move();
            if(!second.waitForMove()) {
                emit resultReady(Sprt::NoResult, m_gpu, m_game, 0);
                return;
            }
            second.readMove();
            first.setMove(wmove + second.getMove());          
            second.nextMove();
        } while (first.nextMove());
        QTextStream(stdout) << "Game has ended." << endl;
        int result = 0;
        if (first.getScore()) {
            result = first.getWinner();
            first.writeSgf();
            second.writeSgf();
            first.dumpTraining();
        }
        QTextStream(stdout) << "Stopping engine." << endl;
        first.gameQuit();
        second.gameQuit();
        if(result == m_expected)
            emit resultReady(Sprt::Win, m_gpu, m_game, m_expected);
        else
            emit resultReady(Sprt::Loss, m_gpu, m_game, m_expected);
}

void WorkerThread::init(const int &gpu, const int &game, const QString &gpuIndex) {
     m_gpu = gpu;
     m_game = game;
     m_option = " -g -q  -r 0 -w ";
     if(!gpuIndex.isEmpty())
         m_option.append(" -gpu=" + gpuIndex);
 }

 void WorkerThread::prepare(const QString &firstNet, const QString &secondNet, const int &expected) {
     m_firstNet = firstNet;
     m_secondNet = secondNet;
     m_expected = expected;
 }


Validation::Validation(const int &gpus, const int &games, const QStringList &gpuslist,
                       const QString &firstNet, const QString &secondNet, QMutex *mutex) :
    m_mutex(mutex),
    m_gamesThreads(games*gpus),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_gamesPlayed(0),
    m_firstNet(firstNet),
    m_secondNet(secondNet)
{
}

Validation::~Validation() {
}

void Validation::startGames() { 
    m_mutex->lock();
    QString n1, n2;
    int expected;

    for(int gpu = 0; gpu < m_gpus; ++gpu) {
        for(int game = 0; game < m_games; ++game) {
            connect(&m_gamesThreads[gpu * m_games + game], &WorkerThread::resultReady, this, &Validation::getResult);
            if(m_gpusList.isEmpty()) {
                m_gamesThreads[gpu * m_games + game].init(gpu, game, "");
            }
            else {
                m_gamesThreads[gpu * m_games + game].init(gpu, game, m_gpusList.at(gpu));
            }
            if(game % 2) {
                n1 = m_firstNet;
                n2 = m_secondNet;
                expected = Game::BLACK;
            }
            else {
                n1 = m_secondNet;
                n2 = m_firstNet;
                expected = Game::WHITE;
            }
            m_gamesThreads[gpu * m_games + game].prepare(n1, n2, expected);
            m_gamesThreads[gpu * m_games + game].start();
        }
    }
}

void Validation::getResult(const Sprt::GameResult &result, const int &gpu, const int &game, const int &exp) {
    if(result == Sprt::NoResult) {
        QTextStream(stdout) << "Engine Error." << endl;
        //try to restert
        m_gamesThreads[gpu * m_games + game].start();
        return;
    }
    m_gamesPlayed++;
    m_statistic.addGameResult(result);
    Sprt::Status status = m_statistic.status();
    if(status.result != Sprt::Continue) {
        quitThreads();
        QTextStream(stdout) << "The first net is ";
        QTextStream(stdout) <<  ((status.result ==  Sprt::AcceptH0) ? "wrost " : "better ");
        QTextStream(stdout) << "then the second" << endl;
        m_mutex->unlock();
    }
    else {
        QTextStream(stdout) << m_gamesPlayed << " games played." << endl;
        QTextStream(stdout) << "Status: " << status.result << " Log-Likelyhood Ratio ";
        QTextStream(stdout) << status.llr <<  " Lower Bound " << status.lBound ;
        QTextStream(stdout) << " Upper Bound " << status.uBound << endl;
        QString n1, n2;
        int expected;
        if(exp == Game::BLACK) {
            n1 = m_secondNet;
            n2 = m_firstNet;
            expected = Game::WHITE;
        }
        else {
            n1 = m_firstNet;
            n2 = m_secondNet;
            expected = Game::BLACK;
        }
        m_gamesThreads[gpu * m_games + game].prepare(n1, n2, expected);
        m_gamesThreads[gpu * m_games + game].start();
    }
}

void Validation::quitThreads() {
    for(int gpu = 0; gpu < m_gpus* m_games; ++gpu) {
        m_gamesThreads[gpu].quit();
    }
}

