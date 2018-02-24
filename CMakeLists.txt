# This file is part of Leela Zero.
# Copyright (C) 2017 Marco Calignano
# Copyright (C) 2017 Gian-Carlo Pascutto
# Leela Zero is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# Leela Zero is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

PROJECT(leelaz)
add_subdirectory(gtest EXCLUDE_FROM_ALL) # We don't want to install gtest, exclude it from `all`

# Required Packages
SET(Boost_MIN_VERSION "1.58.0")
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost 1.58.0 REQUIRED program_options)
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
# We need OpenBLAS for now, because we make some specific
# calls. Ideally we'd use OpenBLAS is possible and fall back to
# not doing those calls if it's not present.
if(NOT APPLE)
  set(BLA_VENDOR OpenBLAS)
endif()
FIND_PACKAGE(BLAS REQUIRED)
find_path(BLAS_INCLUDE_DIRS openblas_config.h
  /usr/include
  /usr/local/include
  /usr/include/openblas
  /opt/OpenBLAS/include
  /usr/include/x86_64-linux-gnu
  $ENV{BLAS_HOME}/include)
FIND_PACKAGE(Qt5Core)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# See if we can set optimization flags as expected.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  SET(GccSpecificFlags 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  SET(GccSpecificFlags 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  SET(GccSpecificFlags 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  SET(GccSpecificFlags 0)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  SET(GccSpecificFlags 0)
endif()

IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RELEASE)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

if (GccSpecificFlags)
  SET(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -flto -march=native")
  SET(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-maybe-uninitialized")
  SET(GCC_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS}")
  SET(CMAKE_CXX_FLAGS_DEBUG "${GCC_FLAGS} -g -Og")
  SET(CMAKE_CXX_FLAGS_RELEASE "${GCC_FLAGS} -g -O3 -DNDEBUG")
  SET(CMAKE_EXE_LINKER_FLAGS "-flto -g")
endif(GccSpecificFlags)

if (FEATURE_USE_CPU_ONLY)
  ADD_DEFINITIONS(-DFEATURE_USE_CPU_ONLY)
endif()

SET(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
SET(SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

INCLUDE_DIRECTORIES(${IncludePath})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})

if((UNIX AND NOT APPLE) OR WIN32)
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIRS})
endif()
if(APPLE)
    INCLUDE_DIRECTORIES("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
endif()

SET(leelaz_MAIN "${SrcPath}/Leela.cpp")
FILE(GLOB leelaz_SRC "${SrcPath}/*.cpp")
LIST(REMOVE_ITEM leelaz_SRC ${leelaz_MAIN})

# Reuse for leelaz and gtest
ADD_LIBRARY(objs OBJECT ${leelaz_SRC})

ADD_EXECUTABLE(leelaz $<TARGET_OBJECTS:objs> ${leelaz_MAIN})

TARGET_LINK_LIBRARIES(leelaz ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${BLAS_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${OpenCL_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${ZLIB_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${CMAKE_THREAD_LIBS_INIT})
INSTALL(TARGETS leelaz DESTINATION bin)

if (Qt5Core_FOUND)
    if (NOT Qt5Core_VERSION VERSION_LESS "5.3.0")
        ADD_SUBDIRECTORY(autogtp)
        ADD_SUBDIRECTORY(validation)
    else()
        MESSAGE(WARNING "Qt ${Qt5Core_VERSION} is found but does not met required version 5.3.0, \
        build target for `autogtp` and `validation` is disabled.")
    endif()
else()
    MESSAGE(WARNNING "Qt is not found, build for `autogtp` and `validation` is disabled")
endif()

# Google Test below
FILE(GLOB tests_SRC "${SrcPath}/tests/*.cpp")

ADD_EXECUTABLE(tests ${tests_SRC} $<TARGET_OBJECTS:objs>)
if (GccSpecificFlags)
  target_compile_options(tests PRIVATE "-Wno-unused-variable")
endif()

TARGET_LINK_LIBRARIES(tests ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(tests ${BLAS_LIBRARIES})
TARGET_LINK_LIBRARIES(tests ${OpenCL_LIBRARIES})
TARGET_LINK_LIBRARIES(tests ${ZLIB_LIBRARIES})
TARGET_LINK_LIBRARIES(tests gtest_main ${CMAKE_THREAD_LIBS_INIT})
