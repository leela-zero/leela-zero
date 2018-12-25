# autogtp

This is a self-play tool for Leela-Zero. When launched, it will fetch the
best network from the server so far, play a game against itself, and upload
the SGF and training data at the end of the game.

# Requirements

* Qt 5.3 or later with qmake
* C++14 capable compiler
* curl
* gzip and gunzip

## Compiling - Ubuntu and similar, macOS

Follow main page instructions to compile leelaz and autogtp binaries at the same time.

For specific use, you can also compile the autogtp binary separately with qmake.

## Compiling under Visual Studio - Windows

You have to download and install Qt and Qt VS Tools. You only need QtCore to
run. Locate a copy of curl.exe and gzip.exe (a previous Leela release package
will contain them) and put them into the msvc subdir.

Loading leela-zero2015.sln or leela-zero2017.sln will then load this project
and should compile. The two exes (curl.exe and gzip.exe) will also be copied to
the output folder after the build, making it possible to run autogtp.exe
directly.

# Running

autogtp and leelaz binaries need to be in the same directory.

While autogtp is running, typing q+Enter will save the processed data and exit. 
When autogtp runs next, autogtp will continue the game.
