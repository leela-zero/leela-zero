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

class Result {
public:
    enum Type {
        File = 0,
        Win,
        Loss,
        Error
    };
    Result() = default;
    Result(int t, QString n = "") { m_type = t, m_name = n; }
    ~Result() = default;
    void type(int t) { m_type = t; }
    int type() { return m_type; }
    void name(const QString &n) { m_name = n; }
    QString name() { return m_name; }
private:
    int m_type;
    QString m_name;
};

#endif // RESULT_H
