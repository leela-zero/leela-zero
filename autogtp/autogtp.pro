QT       -= gui

TARGET = autogtp
CONFIG   += c++14
CONFIG   += warn_on
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    Game.cpp \
    sprt.cpp \
    validation.cpp \
    production.cpp

HEADERS += \
    Game.h \
    sprt.h \
    validation.h \
    production.h
