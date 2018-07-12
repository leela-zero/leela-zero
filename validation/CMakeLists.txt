
cmake_minimum_required(VERSION 3.1)

add_executable(leelaz-validation
        main.cpp ../autogtp/Game.cpp SPRT.cpp Validation.cpp Results.cpp
        ../autogtp/Game.h SPRT.h Validation.h Results.h ../autogtp/Console.h)
set_target_properties(leelaz-validation PROPERTIES AUTOMOC 1)
target_link_libraries(leelaz-validation Qt5::Core)

install(TARGETS leelaz-validation DESTINATION ${CMAKE_INSTALL_BINDIR})
