#ifndef KEYPRESS_H
#define KEYPRESS_H

#include <QObject>
#include "Management.h"

class KeyPress : public QObject
{
    Q_OBJECT
public:
    explicit KeyPress(Management *boss, QObject *parent = nullptr);

protected:
    bool eventFilter(QObject *obj, QEvent *event);
    Management *m_boss;
};

#endif // KEYPRESS_H

