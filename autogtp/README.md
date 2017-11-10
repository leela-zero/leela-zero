# autogtp

This is a self-play tool for Leela-Zero. When launched, it will fetch the
best network from the server so far, play a game against itself, and upload
the SGF and training data at the end of the game.

# Requirements

* Qt 4.x or 5.x with qmake
* C++14 capable compiler
* curl
* gzip and gunzip

# Compiling

Run:

    qmake
    make
