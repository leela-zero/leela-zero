
CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

ADD_EXECUTABLE(autogtp
	Game.h Order.h Management.h Worker.h Job.h Result.h
	Worker.cpp Management.cpp Job.cpp main.cpp Game.cpp)
SET_TARGET_PROPERTIES(autogtp PROPERTIES AUTOMOC 1)
TARGET_LINK_LIBRARIES(autogtp Qt5::Core)

INSTALL(TARGETS autogtp DESTINATION bin)