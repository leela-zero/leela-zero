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

#ifndef ORDER_H
#define ORDER_H

#include <QString>
#include <QMap>

class Order {
public:
    enum {
        Error = 0,
        Production,
        Validation
    };
    Order() = default;
    Order(int t, QMap<QString,QString> p = QMap<QString,QString>()) { m_type = t; m_parameters = p; }
    Order(const Order &o) { m_type = o.m_type; m_parameters = o.m_parameters; }
    Order &operator=(const Order &o) { m_type = o.m_type; m_parameters = o.m_parameters; return *this; }
    ~Order() = default;
    void type(int t) { m_type = t; }
    int type() { return m_type; }
    QMap<QString,QString> parameters() { return m_parameters; }
    void parameters(const QMap<QString,QString> &l) { m_parameters = l; }
    bool isValid() { return (m_type == Production || m_type == Validation); }
private:
    int m_type;
    QMap<QString,QString> m_parameters;
};

#endif // ORDER_H
