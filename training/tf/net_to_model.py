#!/usr/bin/env python3
import os
import sys
import gzip
from tfprocess import TFProcess

with gzip.open(sys.argv[1], 'rb') as f:
    weights = []
    for e, line in enumerate(f):
        line = line.decode('utf-8')
        if e == 0:
            #Version
            print("Version", line.strip())
            if line != '3\n':
                raise ValueError("Unknown version {}".format(line.strip()))
        else:
            weights.append(list(map(float, line.split(' '))))
        if e == 2:
            channels = len(line.split(' '))
            print("Channels", channels)

    blocks = e - (5 + 14)
    if blocks % 14 != 0:
        raise ValueError("Inconsistent number of weights in the file")
    blocks //= 14
    print("Blocks", blocks)

tfprocess = TFProcess(blocks, channels)
tfprocess.init(batch_size=1, gpus_num=1)
tfprocess.replace_weights(weights)
path = os.path.join(os.getcwd(), "leelaz-model")
save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)
