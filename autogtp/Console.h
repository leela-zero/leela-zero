#ifndef CONSOLE_H
#define CONSOLE_H

#include <QObject>
#include <QSocketNotifier>
#include <QTextStream>
#include "stdio.h"


#ifdef Q_OS_WIN
    #include <QWinEventNotifier>
    #include <windows.h>
    typedef QWinEventNotifier Notifier;
#else
    #include <QSocketNotifier>
    typedef QSocketNotifier Notifier;
#endif


class Console : public QObject
{
    Q_OBJECT
public:
    Console(QObject *parent = nullptr)
        : QObject(parent),
#ifdef Q_OS_WIN
          m_notifier(GetStdHandle(STD_INPUT_HANDLE)) {
#else
          m_notifier(fileno(stdin), Notifier::Read) {
#endif
            connect(&m_notifier, &Notifier::activated, this, &Console::readInput);
        }
    ~Console() = default;

signals:
    void sendQuit();

public slots:
    void readInput() {
        QTextStream qin(stdin);
        QString line = qin.readLine();
        if (line.contains("q")) {
            emit sendQuit();
        }
    }
private:
    Notifier m_notifier;
};

#endif // CONSOLE_H
