FROM leela-zero:base

# CPU build
RUN CXX=g++ CC=gcc cmake -DUSE_CPU_ONLY=1 ..

CMD cmake --build . --target leelaz --config Release -- -j2
