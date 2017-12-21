QT_REQ_VERSION = 5.3

lessThan(QT_VERSION, $$QT_REQ_VERSION) {
    error(Minimum supported Qt5 version is $$QT_REQ_VERSION!)
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
