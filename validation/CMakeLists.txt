
cmake_minimum_required(VERSION 3.1)

add_executable(validation
        main.cpp ../autogtp/Game.cpp SPRT.cpp Validation.cpp Results.cpp
        ../autogtp/Game.h SPRT.h Validation.h Results.h ../autogtp/Console.h)
set_target_properties(validation PROPERTIES AUTOMOC 1)
target_link_libraries(validation Qt5::Core)

install(TARGETS validation DESTINATION bin)
