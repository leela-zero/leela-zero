import os
import sys

import threading

num = int(sys.argv[1])
for i in range(num):
    os.system(" ./autogtp &>/dev/null &")
    os.system("sleep 0.1")

