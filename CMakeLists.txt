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
add_subdirectory(gtest)

# Required Packages
SET(Boost_MIN_VERSION "1.58.0")
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost 1.58.0 REQUIRED program_options)
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(Qt5Core REQUIRED)
if (Qt5Core_VERSION VERSION_LESS 5.3.0)
    message(FATAL_ERROR "Minimum supported Qt5 version is 5.3.0!")
endif()

MESSAGE( STATUS "Qt5Core version: " ${Qt5Core_VERSION} )
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

if (GccSpecificFlags)
  SET(CMAKE_CXX_FLAGS "-Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14 -DNDEBUG")
  SET(CMAKE_EXE_LINKER_FLAGS "-flto -g")
endif(GccSpecificFlags)

SET(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
SET(SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

INCLUDE_DIRECTORIES(${IncludePath})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})

if(UNIX AND NOT APPLE)
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIRS})
endif()
if(APPLE)
    INCLUDE_DIRECTORIES("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
endif()

FILE(GLOB leelaz_SRC "${SrcPath}/*.cpp")

ADD_EXECUTABLE(leelaz ${leelaz_SRC})

TARGET_LINK_LIBRARIES(leelaz ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${BLAS_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${OpenCL_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${ZLIB_LIBRARIES})
TARGET_LINK_LIBRARIES(leelaz ${CMAKE_THREAD_LIBS_INIT})

# Google Test below
# `Leela.cpp` needs to be removed since it has main entry
FILE(GLOB tests_SRC "${SrcPath}/tests/*.cpp")
list(REMOVE_ITEM leelaz_SRC "${SrcPath}/Leela.cpp")

ADD_EXECUTABLE(tests ${tests_SRC} ${leelaz_SRC})

TARGET_LINK_LIBRARIES(tests ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(tests ${BLAS_LIBRARIES})
TARGET_LINK_LIBRARIES(tests ${OpenCL_LIBRARIES})
TARGET_LINK_LIBRARIES(tests ${ZLIB_LIBRARIES})
TARGET_LINK_LIBRARIES(tests gtest_main ${CMAKE_THREAD_LIBS_INIT})
