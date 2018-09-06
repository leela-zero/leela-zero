
cmake_minimum_required(VERSION 3.1)

add_executable(autogtp
	Game.h Order.h Management.h Worker.h Job.h Result.h Console.h
	Worker.cpp Management.cpp Job.cpp main.cpp Game.cpp Order.cpp)
set_target_properties(autogtp PROPERTIES AUTOMOC 1)
target_link_libraries(autogtp Qt5::Core)

install(TARGETS autogtp DESTINATION ${CMAKE_INSTALL_BINDIR})
