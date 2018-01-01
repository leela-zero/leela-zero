/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Marco Calignano

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

#ifndef RESULT_H
#define RESULT_H

#include <QString>
#include <QMap>

class Result {
public:
    enum Type {
        File = 0,
        Win,
        Loss,
        Error
    };
    Result() = default;
    Result(int t, QMap<QString,QString> n = QMap<QString,QString>()) { m_type = t, m_parameters = n; }
    ~Result() = default;
    void type(int t) { m_type = t; }
    int type() { return m_type; }
    void add(const QString &name, const QString &value) { m_parameters[name] = value; }
    QMap<QString,QString> parameters() { return m_parameters; }
    void clear() { m_parameters.clear(); }
private:
    int m_type;
    QMap<QString,QString> m_parameters;
};

#endif // RESULT_H
