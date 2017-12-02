QT       -= gui

TARGET = autogtp
CONFIG   += c++14
CONFIG   += warn_on
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    Game.cpp \
    SPRT.cpp \
    Validation.cpp \
    Distributedval.cpp \
    Worker.cpp \
    Managment.cpp

HEADERS += \
    Game.h \
    SPRT.h \
    Validation.h \
   Distributedval.h \
    Worker.h \
    Managment.h
