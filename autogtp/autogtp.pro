QT_REQ_MAJOR_VERSION = 5
QT_REQ_MINOR_VERSION = 3
QT_REQ_VERSION = "$$QT_REQ_MAJOR_VERSION"."$$QT_REQ_MINOR_VERSION"

lessThan(QT_MAJOR_VERSION, $$QT_REQ_MAJOR_VERSION) {
    error(Minimum supported Qt version is $$QT_REQ_VERSION!)
}
equals(QT_MAJOR_VERSION, $$QT_REQ_MAJOR_VERSION):lessThan(QT_MINOR_VERSION, $$QT_REQ_MINOR_VERSION) {
    error(Minimum supported Qt version is $$QT_REQ_VERSION!)
}

QT  -= gui

TARGET = autogtp
CONFIG   += c++14
CONFIG   += warn_on
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    Game.cpp \
    Worker.cpp \
    Job.cpp \
    Management.cpp

HEADERS += \
    Game.h \
    Worker.h \
    Job.h \
    Order.h \
    Result.h \
    Management.h
