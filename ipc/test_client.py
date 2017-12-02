import time
import mmap
import os
import sys
import numpy as np

import posix_ipc as ipc

bsize = int(sys.argv[1])

name = "smleela"
bs   = 4*18*19*19

sm = ipc.SharedMemory( name, flags = ipc.O_CREAT, size = bs*bsize + 8 )
smout = ipc.SharedMemory( "lout", flags = ipc.O_CREAT, size = bs*(19*19+2)*4 )

smp_counter = ipc.Semaphore("lee_counter", ipc.O_CREAT)


mem = mmap.mmap(sm.fd, sm.size)
mv  = memoryview(mem)
counter = mv[0:8]
inp     = mv[8:]

memout = mmap.mmap(smout.fd, smout.size)

smout.close_fd()
sm.close_fd()


myid    = -1

smp_counter.acquire()
myid = counter[0]
counter[0] = counter[0] + 1
smp_counter.release()
print("myid %d" % myid)

smpA= ipc.Semaphore("lee_A_%d" % myid, ipc.O_CREAT)
smpB= ipc.Semaphore("lee_B_%d" % myid, ipc.O_CREAT)
smpB.release()

# memory layout of sm:
# counter |  ....... | ....... | ....... |
#

board = np.zeros ( bs // 4, dtype=np.float32)
myboard = inp[myid*bs:myid*bs + bs]

oz = 19*19 + 2
myout   = memoryview(memout)[myid*4*oz:myid*4*oz + 4*oz]
npout = np.zeros( oz, dtype=np.float32)

for i in range(1000):
    # looping forever
    board[:] = float(i)

    # ok free to copy data to shared memory
    # print(len(myboard), len(board))
    myboard[:] = board.view(dtype=np.uint8)
    smpB.release()

    smpA.acquire()
    npout = np.frombuffer(myout, dtype=np.float32, count=oz)
    # do what ever here
