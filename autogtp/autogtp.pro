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
    Managment.cpp \
    Job.cpp

HEADERS += \
    Game.h \
    Worker.h \
    Managment.h \
    Job.h \
    Order.h \
    Result.h
