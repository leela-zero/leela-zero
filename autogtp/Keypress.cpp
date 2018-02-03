#include "Keypress.h"
#include <QtGui/QKeyEvent>

KeyPress::KeyPress(Management *boss, QObject *parent) :
    QObject(parent),
    m_boss(boss) {
}

bool KeyPress::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::KeyPress) {
        QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
        if (keyEvent->modifiers() == Qt::ControlModifier &&
            keyEvent->key() == Qt::Key_C ) {
            m_boss->storeGames();
        }
        qDebug("Ate key press %d", keyEvent->key());
        return true;
    } else {
        // standard event processing
        return QObject::eventFilter(obj, event);
    }
}
