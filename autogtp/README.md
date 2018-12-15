# autogtp

This is a self-play tool for Leela-Zero. When launched, it will fetch the
best network from the server so far, play a game against itself, and upload
the SGF and training data at the end of the game.

To work successfully, autogtp binary needs to also have leelaz binary to be in the same directory.

# Requirements

* Qt 5.3 or later with qmake
* C++14 capable compiler
* curl
* gzip and gunzip

## Compiling - Ubuntu and similar

- To compile both autogtp and leelaz binaries at the same time, see : [Example of compiling (Ubuntu and similar)](https://github.com/gcp/leela-zero/tree/next#example-of-compiling---ubuntu--similar)

- However, if for some reason you want to only compile autogtp binary (without leelaz), you can follow these instructions instead, leela-zero/autogtp/ directory : 

      sudo apt install qt5-default qt5-qmake curl
      qmake -qt5
      make
      
If you compile autogtp this way, remember to copy leelaz binary to the directory where autogtp binary is. For example, if leelaz binary has been compiled in leela-zero/build/, you can do it like that : 

      cp ../build/leelaz .
      
Then run autogtp to start contributing : 

      ./autogtp

## Compiling - macOS

- To compile both autogtp and leelaz binaries at the same time, see : [Example of compiling (macOS)](https://github.com/gcp/leela-zero/tree/next#example-of-compiling---macos)

- However, if for some reason you want to only compile autogtp binary (without leelaz), you can follow these instructions instead : 

??? is empty in gcp/master and gcp/next ??? https://github.com/gcp/leela-zero/tree/next/autogtp

## Compiling under Visual Studio - Windows

See : [Example of compiling (Windows)](https://github.com/gcp/leela-zero/tree/next#example-of-compiling---windows)

You have to download and install Qt and Qt VS Tools. You only need QtCore to
run. Locate a copy of curl.exe and gzip.exe (a previous Leela release package
will contain them) and put them into the msvc subdir.

Loading leela-zero2015.sln or leela-zero2017.sln will then load this project
and should compile. The two exes (curl.exe and gzip.exe) will also be copied to
the output folder after the build, making it possible to run autogtp.exe
directly.

# Running

See : [Contributing](https://github.com/gcp/leela-zero/tree/next#contributing)

While autogtp is running, typing q+Enter will save the processed data and exit. When autogtp runs next, autogtp will continue the game.
