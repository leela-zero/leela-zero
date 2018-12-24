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

#include "Order.h"
#include <QFile>
#include <QTextStream>

void Order::save(const QString &file) {
    QFile f(file);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return;
    }
    QTextStream out(&f);
    out << m_type << endl;
    out << m_parameters.size() << endl;
    for (QString key : m_parameters.keys())
    {
        out << key << " " << m_parameters.value(key) << endl;
    }
    out.flush();
    f.close();
}

void Order::load(const QString &file) {
    QFile f(file);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return;
    }
    QTextStream in(&f);
    in >>  m_type;
    int count;
    in >> count;
    QString key;
    for (int i = 0; i < count; i++) {
        in >> key;
        if (key == "options" || key == "optionsSecond") {
            m_parameters[key] = in.readLine();
        } else {
            in >> m_parameters[key];
        }
    }
    f.close();
}
