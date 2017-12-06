# autogtp

This is a self-play tool for Leela-Zero. When launched, it will fetch the
best network from the server so far, play a game against itself, and upload
the SGF and training data at the end of the game.

# Requirements

* Qt 5.x with qmake
* C++14 capable compiler
* curl
* gzip and gunzip

## Example of compiling - Ubuntu

    sudo apt install qt5-default qt5-qmake curl
    qmake -qt5
    make

## Compiling under Visual Studio - Windows

You have to download and install Qt and Qt VS Tools. You only need QtCore to run.
Loading leela-zero2015.sln/leela-zero2017.sln will then load this project and 
should compiling. Two exes (curl.exe and gzip.exe) will also be copied to the output
folder after build, making it able to run autogtp.exe directally.

# Running

Copy the compiled leelaz binary into the autogtp directory, and run
autogtp.

    cp ../src/leelaz .
    ./autogtp

