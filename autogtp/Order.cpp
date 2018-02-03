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
    for(QString key : m_parameters.keys())
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
    for(int i = 0; i < count; i++) {
        in >> key;
        if(key == "options") {
           m_parameters[key] = in.readLine();
        } else {
            in >> m_parameters[key];
        }
    }
    f.close();
}
