import time
import mmap
import os
import sys
import numpy as np

import posix_ipc as ipc


name = "smleela"
bs   = 4*18*19*19

sm = ipc.SharedMemory(name, flags = ipc.O_CREAT, size = bs+1)
smp= ipc.Semaphore("smpleela", ipc.O_CREAT)
smp.release()

mem = mmap.mmap(sm.fd, sm.size)
mem[0] = 0

sm.close_fd()

dt = np.zeros( bs ).astype(np.int8)

dt[:] = 1

print(len( dt.astype(np.int8).data) )
print(len(mem))
for i in range(1000000):
    while True:
        smp.release()
        # nothing here
        smp.acquire()
        if mem[0]  > 0:
            break

    if mem[0] == 2:
        break

    mem[0] = 0
    # okay cool!
    dt[:] = np.frombuffer(mem, dtype=np.int8, count=bs, offset=1)


smp.release()

mem[0] = 10

mem.close()
ipc.unlink_shared_memory(name)
smp.unlink()
