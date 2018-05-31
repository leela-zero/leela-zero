/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Marco Calignano

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

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
