import time
import mmap
import os
import sys
import numpy as np

import posix_ipc as ipc

if len(sys.argv) != 3 :
    print("Usage: %s num-instances batch-size" % sys.argv[0])
    sys.exit(-1)

num_instances = int(sys.argv[1])
realbs = int(sys.argv[2])

if num_instances % realbs != 0:
    print("ERRRRRR")
    sys.exit(-1)

name = "smlee"
bs   = 4*18*19*19
def createSMP(name):
    smp= ipc.Semaphore(name, ipc.O_CREAT)
    smp.unlink()
    return ipc.Semaphore(name, ipc.O_CREAT)

sm = ipc.SharedMemory( name, flags = ipc.O_CREAT, size = 2 + num_instances + bs*num_instances + 8  + num_instances*4*(19*19+2))

smp_counter =  createSMP("lee_counter")


smpA = []
smpB = []
for i in range(num_instances):
    smpA.append(createSMP("lee_A_%d" % i))
    smpB.append(createSMP("lee_B_%d" % i))

# memory layout of sm:
# counter |  ....... | ....... | ....... |
#

mem = mmap.mmap(sm.fd, sm.size)
sm.close_fd()

mv  = np.frombuffer(mem, dtype=np.uint8, count= 2 + num_instances + bs*num_instances + 8  + num_instances*4*(19*19+2))
counter = mv[0:2+num_instances]
inp     = mv[  2+num_instances:2+num_instances + bs*num_instances]
memout =  mv[                  2+num_instances + bs*num_instances + 8:]

import nn

counter[0] = num_instances // 256
counter[1] = num_instances %  256

for i in range(num_instances):
    counter[2 + i ] = 0

smp_counter.release()

# waiting clients to connect
print("Waiting for %d autogtp instances to run" % num_instances)
# for i in range(bsize):
#     smpB[i].acquire()
#
# print("OK Go!")

# now all clients connected
# dt = np.zeros( bs*bsize // 4, dtype=np.float32)

net = nn.net
import gc
import time

t2 = time.perf_counter()
numiter = num_instances // realbs
outsize = 4 * (19*19+2)

while True:
    for ii in range(numiter):
        start = ii * realbs
        # print(c)

        # wait for data
        for i in range(realbs):
            smpB[start + i].acquire()

        # t1 = time.perf_counter()
        # print("delta t1 = ", t1 - t2)
        # t1 = time.perf_counter()


        dt = np.frombuffer(inp[(start+ 0)*bs: (start+ realbs)*bs], dtype=np.float32, count=bs*realbs // 4)

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


        net[0].set_value(dt.reshape( (realbs, 18, 19, 19) ) )

        qqq = net[1]().astype(np.float32)
        ttt = qqq.reshape(realbs * (19*19+2))
        #print(len(ttt)*4, len(memout))
        memout[ii*outsize:(ii+realbs)*outsize] = ttt.view(dtype=np.uint8)

        for i in range(realbs):
            smpA[start+i].release() # send result to client
        # t2 = time.perf_counter()
        # print("delta t2 = ", t2- t1)
        # t2 = time.perf_counter()

