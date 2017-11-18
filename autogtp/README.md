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

# Running

Copy the compiled leelaz binary into the autogtp directory, and run
autogtp.

    cp ../src/leelaz .
    ./autogtp

