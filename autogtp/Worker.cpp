#include "Worker.h"
#include "Game.h"
#include <QTextStream>
#include <chrono>


Worker::Worker(int index,const QString& gpuIndex,const QString& keep) :
    m_index(index),
    m_state(),
    m_keepPath(keep),
    m_option(""),
    m_job(nullptr)
{
    if (!gpuIndex.isEmpty()) {
        m_option = " --gpu=" + gpuIndex + " ";
    }
}

void Worker::order(Order o)
{
    if(m_todo.type() != o.type() || m_job == nullptr) {
        createJob(o.type());
    }
     m_todo = o;
     m_job->init(m_todo.parameters());
}


void Worker::createJob(int type) {
    if(m_job != nullptr) {
        delete m_job;
    }
    switch(type) {
    case Order::Production:
        m_job = new ProdutionJob();
        break;
    case Order::Validation:
        m_job = new ValidationJob();
        break;
    default:
        m_job = nullptr;
    }
}

void Worker::run() {
     do {
        auto start = std::chrono::high_resolution_clock::now();

        Result res = m_job->execute();

        auto end = std::chrono::high_resolution_clock::now();
        auto gameDuration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        emit resultReady(m_todo, res, m_index, gameDuration);
    } while (m_state != FINISHING);
    QTextStream(stdout) << "Program ends: exiting." << endl;
}
