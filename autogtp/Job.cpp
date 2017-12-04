#include "Job.h"
#include "Game.h"
#include <QTextStream>
#include <chrono>

#include <QThread>

Job::Job() :
  m_state(RUNNING),
  m_option("")
{
}

ProdutionJob::ProdutionJob() :
Job()
{
}

ValidationJob::ValidationJob() :
Job()
{
}

Result ProdutionJob::execute(){
    Result res(Result::Error);
    Game game(m_network, m_option);
    if(!game.gameStart(min_leelaz_version)) {
        return res;
    }
    do {
        game.move();
        if(!game.waitForMove()) {
            return res;
        }
        game.readMove();
    } while (game.nextMove() && m_state.load() == RUNNING);
    if(m_state.load() == RUNNING) {
        QTextStream(stdout) << "Game has ended." << endl;
        if (game.getScore()) {
            game.writeSgf();
            game.dumpTraining();
        }
        res.type(Result::File);
        res.name(game.getFile());
    } else {
        QTextStream(stdout) << "Program ends: exiting." << endl;
    }
    QTextStream(stdout) << "Stopping engine." << endl;
    game.gameQuit();
    return res;
}

void ProdutionJob::init(const QStringList &l) {
    Job::init(l);
    m_network = l[1];
}


Result ValidationJob::execute(){
   Result res(Result::Error);
   Game first(m_firstNet,  m_option);
   if(!first.gameStart(min_leelaz_version)) {
       return res;
   }
   Game second(m_secondNet, m_option);
   if(!second.gameStart(min_leelaz_version)) {
       return res;
   }
   QString wmove = "play white ";
   QString bmove = "play black ";
   do {
       first.move();
       if(!first.waitForMove()) {
           return res;
       }
       first.readMove();
       second.setMove(bmove + first.getMove());
       second.move();
       if(!second.waitForMove()) {
           return res;
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
       // Game is finished, send the result
       res.type(Result::Win);
       if (result == Game::BLACK) {
           res.name("Black");
       } else {
           res.name("White");
       }
   }
   QTextStream(stdout) << "Stopping engine." << endl;
   first.gameQuit();
   second.gameQuit();
   return res;
}


void ValidationJob::init(const QStringList &l) {
    Job::init(l);
    m_firstNet = l[1];
    m_secondNet = l[2];
}





