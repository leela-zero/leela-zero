FROM leela-zero:base

# GPU build
RUN CXX=g++ CC=gcc cmake ..

CMD cmake --build . --target leelaz --config Release -- -j2
