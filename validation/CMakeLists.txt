
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

ADD_EXECUTABLE(validation
        main.cpp ../autogtp/Game.cpp SPRT.cpp Validation.cpp Results.cpp
        ../autogtp/Game.h SPRT.h Validation.h Results.h ../autogtp/Console.h)
SET_TARGET_PROPERTIES(validation PROPERTIES AUTOMOC 1)
TARGET_LINK_LIBRARIES(validation Qt5::Core)

INSTALL(TARGETS validation DESTINATION bin)