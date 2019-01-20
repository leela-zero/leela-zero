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

# Help 

For more details about AutoGTP, or for specific use (for example how to use multi GPU, 
or any other setting), you can run, while in the directory where your autogtp executable is :
- `./autogtp --help` on linux/mac
- `autogtp.exe --help` on windows

This will display a list of all possible settings and how to use them

For your convenience, and for easier reference, a copy is provided below :

```
Usage: ./autogtp [options]

Options:
  -h, --help                            Displays this help.
  -v, --version                         Displays version information.
  -g, --gamesNum <num>                  Play 'gamesNum' games on one GPU at the
                                        same time.
  -u, --gpus <num>                      Index of the GPU to use for multiple
                                        GPUs support.
  -k, --keepSgf <output directory>      Save SGF files after each self-play
                                        game.
  -d, --debug <output directory>        Save training and extra debug files
                                        after each self-play game.
  -t, --timeout <time in minutes>       Save running games after the timeout
                                        (in minutes) is passed and then exit.
  -s, --single                          Exit after the first game is completed.
  -m, --maxgames <max number of games>  Exit after the given number of games is
                                        completed.
  -e, --erase                           Erase old networks when new ones are
                                        available.

```
