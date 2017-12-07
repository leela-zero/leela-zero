QT  -= gui

TARGET = validation
CONFIG   += c++14
CONFIG   += warn_on
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    ../autogtp/Game.cpp \
    SPRT.cpp \
    Validation.cpp \
    Results.cpp

HEADERS += \
    ../autogtp/Game.h \
    SPRT.h \
    Validation.h \
    Results.h
