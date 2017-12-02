import time
import mmap
import os
import sys
import numpy as np

import posix_ipc as ipc

if len(sys.argv) != 2 :
    print("Usage: %s batch-size" % sys.argv[0])
    sys.exit(-1)

bsize = int(sys.argv[1])

name = "smlee"
bs   = 4*18*19*19
def createSMP(name):
    smp= ipc.Semaphore(name, ipc.O_CREAT)
    smp.unlink()
    return ipc.Semaphore(name, ipc.O_CREAT)

sm = ipc.SharedMemory( name, flags = ipc.O_CREAT, size = 8 + bs*bsize + 8  + bsize*4*(19*19+2))

smp_counter =  createSMP("lee_counter")


smpA = []
smpB = []
for i in range(bsize):
    smpA.append(createSMP("lee_A_%d" % i))
    smpB.append(createSMP("lee_B_%d" % i))

# memory layout of sm:
# counter |  ....... | ....... | ....... |
#

mem = mmap.mmap(sm.fd, sm.size)
sm.close_fd()

mv  = memoryview(mem)
counter = mv[0:8]
inp     = mv[8:8+bs*bsize]
memout =  mv[8+bs*bsize + 8:]

import nn

counter[0] = 0
counter[1] = bsize
memout[0] = bsize + 1
smp_counter.release()

# waiting clients to connect
print("Waiting for %d autogtp instances to run" % bsize)
for i in range(bsize):
    smpB[i].acquire()

print("OK Go!")

# now all clients connected
dt = np.zeros( bs*bsize // 4, dtype=np.float32)

net = nn.net
npout = np.zeros ( bsize*(19*19+2) )
import gc

while True:
    # print(c)

    # wait for data
    for i in range(bsize):
        smpB[i].acquire()

    dt[:] = np.frombuffer(inp, dtype=np.float32, count=bs*bsize // 4)

    nn.netlock.acquire(True)   # BLOCK HERE
    if nn.newNetWeight != None:
        nn.net = None
        gc.collect()  # hope that GPU memory is freed, not sure :-()
        weights, numBlocks, numFilters = nn.newNetWeight
        print(" %d channels and %d blocks" % (numFilters, numBlocks) )
        nn.net = nn.LZN(weights, numBlocks, numFilters)
        net = nn.net
        print("...updated weight!")
        nn.newNetWeight = None
    nn.netlock.release()


    net[0].set_value(dt.reshape( (bsize, 18, 19, 19) ) )

    qqq = net[1]().astype(np.float32)
    ttt = qqq.reshape(bsize * (19*19+2))
    #print(len(ttt)*4, len(memout))
    memout[:] = ttt.view(dtype=np.uint8)

    for i in range(bsize):
        smpA[i].release() # send result to client

