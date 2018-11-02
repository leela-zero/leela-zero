# autogtp

This is a self-play tool for Leela-Zero. When launched, it will fetch the
best network from the server so far, play a game against itself, and upload
the SGF and training data at the end of the game.

# Requirements

* cmake
* C++14 capable compiler
* curl
* gzip and gunzip

## How to compile AutoGTP - Ubuntu and macOS

Just follow main instructions : cmake will compile both autogtp and leelaz binaries in build subdirectory.

You don't need to do anything else.


## How to compile AutoGTP using Visual Studio - Windows

You have to download and install Qt and Qt VS Tools. You only need QtCore to
run. Locate a copy of curl.exe and gzip.exe (a previous Leela release package
will contain them) and put them into the msvc subdir.

Loading leela-zero2015.sln or leela-zero2017.sln will then load this project
and should compile. The two exes (curl.exe and gzip.exe) will also be copied to
the output folder after the build, making it possible to run autogtp.exe
directly.

# Running AutoGTP (to start contributing)

As explained in main page instructions, copy compiled leelaz and autogtp binaries (with cmake) from build subdirectory to autogtp subdirectory, then run autogtp to start contributing.

    ./autogtp

While autogtp is running, typing q+Enter will save the processed data and exit. When autogtp runs next, autogtp will continue the game.
