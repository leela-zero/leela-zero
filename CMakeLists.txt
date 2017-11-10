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

CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

PROJECT(leelaz)

# Required Packages

SET(BOOST_MIN_VERSION "1.58.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED program_options)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)

# We need OpenBLAS for now, because we make some specific
# calls. Ideally we'd use OpenBLAS is possible and fall back to
# not doing those calls if it's not present.
set(BLA_VENDOR OpenBLAS)
FIND_PACKAGE(BLAS REQUIRED)
find_path(BLAS_INCLUDE_DIRS openblas_config.h
  /usr/include
  /usr/local/include
  /usr/include/openblas
  /opt/OpenBLAS/include
  /usr/include/x86_64-linux-gnu
  $ENV{BLAS_HOME}/include)

SET(IncludePath "CMAKE_CURRENT_SOURCE_DIR}/src")
SET(SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14 -DNDEBUG")
SET(CMAKE_EXE_LINKER_FLAGS "-flto -g")

INCLUDE_DIRECTORIES(${IncludePath})
FILE(GLOB leelaz_SRC "${SrcPath}/*.cpp")

ADD_EXECUTABLE(leelaz ${leelaz_SRC})

TARGET_LINK_LIBRARIES(leelaz pthread)
if(Boost_FOUND)
  INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${Boost_LIBRARIES})
endif(Boost_FOUND)

if(BLAS_FOUND)
  INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

if(OpenCL_FOUND)
  INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)

if(ZLIB_FOUND)
  INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${ZLIB_LIBRARIES})
endif(ZLIB_FOUND)
