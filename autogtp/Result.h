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
