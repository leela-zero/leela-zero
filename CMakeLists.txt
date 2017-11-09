CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#add_definitions(-DOEM_TYPE=${OEM})

#Required Packages
PROJECT(leelaz)

SET (BOOST_MIN_VERSION "1.58.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED)
FIND_PACKAGE( BLAS REQUIRED )
FIND_PACKAGE( ZLIB REQUIRED )
FIND_PACKAGE( OpenCL REQUIRED )

SET(IncludePath "CMAKE_CURRENT_SOURCE_DIR}/src")
SET(SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

INCLUDE_DIRECTORIES(${IncludePath})
INCLUDE_DIRECTORIES(/usr/include/openblas)

SET(SOURCES
    "${SrcPath}/FastBoard.cpp"
    "${SrcPath}/FastState.cpp"
    "${SrcPath}/FullBoard.cpp"
    "${SrcPath}/GameState.cpp"
    "${SrcPath}/GTP.cpp"
    "${SrcPath}/KoState.cpp"
    "${SrcPath}/Leela.cpp"
    "${SrcPath}/Network.cpp"
    "${SrcPath}/OpenCL.cpp"
    "${SrcPath}/Random.cpp"
    "${SrcPath}/SGFParser.cpp"
    "${SrcPath}/SGFTree.cpp"
    "${SrcPath}/SMP.cpp"
    "${SrcPath}/TimeControl.cpp"
    "${SrcPath}/Timing.cpp"
    "${SrcPath}/Training.cpp"
    "${SrcPath}/TTable.cpp"
    "${SrcPath}/UCTNode.cpp"
    "${SrcPath}/UCTSearch.cpp"
    "${SrcPath}/Utils.cpp"
    "${SrcPath}/Zobrist.cpp"
)

ADD_EXECUTABLE(leelaz ${SOURCES})

TARGET_LINK_LIBRARIES(leelaz openblas)
TARGET_LINK_LIBRARIES(leelaz OpenCL)
TARGET_LINK_LIBRARIES(leelaz pthread)
TARGET_LINK_LIBRARIES(leelaz boost_program_options)
TARGET_LINK_LIBRARIES(leelaz m)
TARGET_LINK_LIBRARIES(leelaz z)

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14 -DNDEBUG")
SET(CMAKE_EXE_LINKER_FLAGS "-flto -g")

