/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Marco Calignano
    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

PROJECT(leelaz)

#Required Packages

SET(BOOST_MIN_VERSION "1.58.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(OPENCL REQUIRED)


if(BOOST_FOUND)
  INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${BOOST_LIBRARIES})
endif(BOOST_FOUND)

if(BLAS_FOUND)
  INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

if(OPENCL_FOUND)
  INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${OPENCL_LIBRARIES})
endif(OPENCL_FOUND)

if(ZLIB_FOUND)
  INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(leelaz ${ZLIB_LIBRARIES})
endif(ZLIB_FOUND)

SET(IncludePath "CMAKE_CURRENT_SOURCE_DIR}/src")
SET(SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

INCLUDE_DIRECTORIES(${IncludePath})

FILE(GLOB leelaz_SRC "${SrcPath}/*.cpp")

ADD_EXECUTABLE(leelaz ${leelaz_SRC})

TARGET_LINK_LIBRARIES(leelaz pthread)

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14 -DNDEBUG")
SET(CMAKE_EXE_LINKER_FLAGS "-flto -g")

