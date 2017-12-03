#ifndef ORDER_H
#define ORDER_H

#include <QStringList>

class Order {
public:
    enum {
        Error = 0,
        Production,
        Validation
    };
    Order() = default;
    Order(int t, QStringList p = QStringList()) { m_type = t; m_parameters = p; }
    ~Order() = default;
    void type(int t) { m_type = t; }
    int type() { return m_type; }
    QStringList parameters() { return m_parameters; }
    void parameters(const QStringList &l) { m_parameters = l; }
private:
    int m_type;
    QStringList m_parameters;
};

#endif // ORDER_H
