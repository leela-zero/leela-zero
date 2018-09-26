FROM leela-zero:base

# GPU build
RUN CXX=g++ CC=gcc cmake -DUSE_BLAS=1 ..

CMD cmake --build . --target leelaz --config Release -- -j2
