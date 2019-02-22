# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindReadLine
# ----------
#
# Try to find GNU readline
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables::
#
#   Readline_FOUND        - True if ReadlineCL was found
#   Readline_INCLUDE_DIRS - include directories for readline
#   Readline_LIBRARIES    - link against this library to use readline
#
# The module will also define two cache variables::
#
#   OpenCL_INCLUDE_DIR    - the OpenCL include directory
#   OpenCL_LIBRARY        - the path to the OpenCL library
#
# Search for the path containing library's headers

# Search for include directory
find_path(Readline_INCLUDE_DIR
    NAMES readline/readline.h
    HINTS ${Readline_ROOT_DIR}/include
)

# Search for library
find_library(Readline_LIBRARY
    NAMES readline
    HINTS ${Readline_ROOT_DIR}/lib
)

set(Readline_LIBRARIES ${Readline_LIBRARY})
set(Readline_INCLUDE_DIRS ${Readline_INCLUDE_DIR})
set(Readline_VERSION_STRING "")


include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(
    Readline
    FOUND_VAR Readline_FOUND
    REQUIRED_VARS Readline_LIBRARY Readline_INCLUDE_DIR
    VERSION_VAR Readline_VERSION_STRING)

MARK_AS_ADVANCED(
    Readline_INCLUDE_DIR
    Readline_LIBRARY)

