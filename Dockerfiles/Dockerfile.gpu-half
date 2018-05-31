FROM leela-zero:base

# GPU 16-bit (memory storage) build
RUN CXX=g++ CC=gcc cmake -DUSE_HALF=1 ..

CMD cmake --build . --target leelaz --config Release -- -j2
