#ifndef CONSOLE_H
#define CONSOLE_H

#include <QObject>
#include <QSocketNotifier>
#include <QTextStream>
#include "stdio.h"

class Console : public QObject
{
    Q_OBJECT
public:
    Console(QObject *parent = nullptr)
        : QObject(parent),
          m_notifier(fileno(stdin), QSocketNotifier::Read) {
            connect(&m_notifier, &QSocketNotifier::activated, this, &Console::readInput);
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
        QTextStream(stdout) << "captured input: " << line << endl;
    }
private:
    QSocketNotifier m_notifier;
};

#endif // CONSOLE_H
