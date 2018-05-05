FROM leela-zero:base

# CPU build
RUN CXX=g++ CC=gcc cmake -DUSE_CPU_ONLY=1 ..
RUN cmake --build . --target tests --config Release -- -j2

CMD ./tests

