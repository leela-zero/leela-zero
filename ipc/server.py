import time
import mmap
import os
import sys
import numpy as np

import posix_ipc as ipc

if len(sys.argv) != 3 :
    print("Usage: %s num-instances batch-size" % sys.argv[0])
    sys.exit(-1)

if "LEELAZ" in os.environ:
    leename= os.environ["LEELAZ"]
else:
    leename = "lee"

print("Using batch name: ", leename)

num_instances = int(sys.argv[1])
realbs = int(sys.argv[2])               # real batch size

if num_instances % realbs != 0:
    print("Error: number of instances isn't divisible by batch size")
    sys.exit(-1)

name = "sm%s" % leename  # shared memory name
input_size   = 4*18*19*19
output_size = 4 * (19*19 + 2)

def createSMP(name):
    smp= ipc.Semaphore(name, ipc.O_CREAT)
    smp.unlink()
    return ipc.Semaphore(name, ipc.O_CREAT)

sm = ipc.SharedMemory( name, flags = ipc.O_CREAT, size = 2 + num_instances + input_size*num_instances + 8  + num_instances*output_size)
# memory layout of the shared memory:
# | counter counter | input 1 | input 2 | .... |  8 bytes | output 1 | output 2| ..... |

smp_counter =  createSMP("%s_counter" % leename) # counter semaphore

smpA = []
smpB = []
for i in range(num_instances):
    smpA.append(createSMP("%s_A_%d" % (leename,i)))    # two semaphores for each instance
    smpB.append(createSMP("%s_B_%d" % (leename,i)))

mem = mmap.mmap(sm.fd, sm.size)
sm.close_fd()

mv  = np.frombuffer(mem, dtype=np.uint8, count= 2 + num_instances + input_size*num_instances + 8  + num_instances*output_size)
counter = mv[0:2+num_instances]
inp     = mv[  2+num_instances:2+num_instances + input_size*num_instances]
memout =  mv[                  2+num_instances + input_size*num_instances + 8:]

import nn # import our neural network

# reset everything
mv[:] = 0

counter[0] = num_instances // 256   # num_instances = counter0 * 256 + counter1
counter[1] = num_instances %  256

smp_counter.release() # now clients can take this semaphore

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

        dt = np.frombuffer(inp[(start+ 0)*input_size: (start+ realbs)*input_size], dtype=np.float32, count=input_size*realbs // 4)

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
        memout[(start+0)*output_size:(start+realbs)*output_size] = ttt.view(dtype=np.uint8)

        for i in range(realbs):
            smpA[start+i].release() # send result to client
        # t2 = time.perf_counter()
        # print("delta t2 = ", t2- t1)
        # t2 = time.perf_counter()

