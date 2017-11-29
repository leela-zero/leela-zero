import sys
import glob
import gzip
import random
import math
import numpy as np
import trollius
import subprocess
import urllib.request


if len(sys.argv) != 2:
    print("Usage: %s [batch-size]" % sys.argv[0])
    sys.exit(-1)

QLEN     = int(sys.argv[1]) # alias: Batch size
# autogtp_exe = sys.argv[3]

print("Leela Zero TCP Neural Net Service")

def getLastestNNHash():
    global nethash
    txt = urllib.request.urlopen("http://zero.sjeng.org/best-network-hash").read().decode()
    net  = txt.split("\n")[0]
    return net

def downloadBestNetworkWeight(hash):
    try:
        return open(hash + ".txt", "r").read()
    except Exception as ex:
        txt = urllib.request.urlopen("http://zero.sjeng.org/networks/best-network").read().decode()
        open(hash + ".txt", "w").write(txt).close()
        return txt
    

def loadWeight(text):

    linecount = 0

    def testShape(s, si):
        t = 1
        for l in s:
            t = t * l
        if t != si:
            print("ERRROR: ", s, t, si)

    FORMAT_VERSION = "1"

    # print("Detecting the number of residual layers...")

    w = text.split("\n")
    linecount = len(w)

    if w[0] != FORMAT_VERSION:
        print("Wrong version")
        sys.exit(-1)

    count = len(w[2].split(" "))
    # print("%d channels..." % count)

    residual_blocks = linecount - (1 + 4 + 14)

    if residual_blocks % 8 != 0:
        print("Inconsistent number of layers.")
        sys.exit(-1)

    residual_blocks = residual_blocks // 8
    # print("%d blocks..." % residual_blocks)

    plain_conv_layers = 1 + (residual_blocks * 2)
    plain_conv_wts = plain_conv_layers * 4


    weights = [ [float(t) for t in l.split(" ")] for l in w[1:] ]
    return (weights, residual_blocks, count)

print("\nLoading lastest network")
nethash = getLastestNNHash()
print("Hash: " + nethash)
print("Downloading weights")
txt     = downloadBestNetworkWeight(nethash)
print("Done!")

weights, numBlocks, numFilters = loadWeight(txt)
print(" %d channels and %d blocks" % (numFilters, numBlocks) )

def LZN(ws, nb, nf):
    # ws: weights
    # nb: number of blocks
    # nf: number of filters

    global wc  # weight counter
    wc = -1

    def loadW():
        global wc
        wc = wc + 1
        return ws[wc]


    def mybn(inp, nf, params, name):
        #mean0 = T.vector(name + "_mean")
        w = np.asarray(loadW(), dtype=np.float32).reshape( (nf) )
        mean0 = shared(w)
        # params.append(In(mean0, value=w))

        #var0  = T.vector(name + "_var")
        w = np.asarray(loadW(), dtype=np.float32).reshape( (nf) )
        var0 = shared(w)
        #params.append(In(var0, value=w))

        bn0   = bn(inp, gamma=T.ones(nf), beta=T.zeros(nf), mean=mean0,
                var=var0, axes = 'spatial', epsilon=1.0000001e-5)

        return bn0

    def myconv(inp, inc, outc, kernel_size, params, name):
        #f0 = T.tensor4(name + '_filter')
        w = np.asarray(loadW(), dtype=np.float32).reshape( (outc, inc, kernel_size, kernel_size) )
        #params.append(In(f0, value=w))
        f0 = shared(w)

        conv0 = conv2d(inp, f0, input_shape=(None, inc, 19, 19),
                border_mode='half', filter_flip=False, filter_shape=(outc,inc,
                    kernel_size,kernel_size))
        b = loadW()  # zero bias
        return conv0

    def residualBlock(inp, nf, params, name):
        conv0 = myconv(inp, nf, nf, 3, params, name + "_conv0")
        bn0   = mybn(conv0, nf, params, name + "_bn0")
        relu0 = relu(bn0)

        conv1 = myconv(relu0, nf, nf, 3, params, name + "_conv1")
        bn1   = mybn(conv1, nf, params, name + "_bn1")

        sum0  = inp + bn1
        out   = relu(sum0)

        return out

    def myfc(inp, insize, outsize, params, name):
        # W0 = T.matrix(name + '_W')
        w = np.asarray(loadW(), dtype=np.float32).reshape( (outsize, insize) ).T
        #params.append(In(W0, value=w))
        W0 = shared(w)

        # b0 = T.vector(name + '_b')
        b = np.asarray(loadW(), dtype=np.float32).reshape( (outsize) )
        # params.append(In(b0, value=b))
        b0 = shared(b)

        out = tensor.dot(inp, W0) + b0
        return out


    params = []
    x = T.tensor4('input')
    params.append(x)
    conv0 = myconv(x, 18, nf, 3, params, "conv0")
    bn0   = mybn(conv0, nf, params, "bn0")

    relu0 = relu(bn0)
    inp = relu0

    for i in range(nb):
        inp = residualBlock(inp, nf, params, "res%d" % (i+1))

    polconv0 = myconv(inp, nf, 2, 1, params, "polconv0")
    polbn0   = mybn(polconv0, 2, params, "polbn0")
    polrelu0 = relu(polbn0)
    polfcinp = polrelu0.flatten(ndim=2)
    polfcout = myfc(polfcinp, 19*19*2, 19*19+1, params, "polfc")

    out = polfcout

    valconv0 = myconv(inp, nf, 1, 1, params, "valconv0")
    valbn0   = mybn(valconv0, 1, params, "valbn0")
    valrelu0 = relu(valbn0)

    valfc0inp = valrelu0.flatten(ndim=2)
    valfc0out = myfc(valfc0inp, 19*19, 256, params, "valfc0")
    valrelu0  = relu(valfc0out)

    valfc1out = myfc(valrelu0, 256, 1, params, "valfc1")
    valout  = valfc1out

    out = T.concatenate( [polfcout, T.tanh(valout)], axis=1 )
    return function(params, out)

print("\nCompling the lastest neural network")

from theano import *
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu
from theano.tensor.nnet.bn import batch_normalization_test as bn

net = LZN(weights, numBlocks, numFilters)

print("Done!")
### For testing purpose
# inp1 = [math.sin(i) for i in range(1*19*19*18) ]
# inp2 = [math.cos(i) for i in range(1*19*19*18) ]
# inp = []
# inp.extend(inp1)
# inp.extend(inp2)
# inp.extend(inp1)
# inp.extend(inp2)
# inpp = np.asarray(inp, dtype=np.float32).reshape( (4,18,19,19) )
# print(net(inpp))
# for i in range(1000):
#     net(inpp)


import socket
import sys

# -- # Create a UDP socket
# -- sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# --
# -- # Bind the socket to the port
# -- server_address = ('127.0.0.1', 9999)
# -- print('starting up on {} port {}'.format(*server_address))
# -- sock.bind(server_address)
# --
# -- inp = np.zeros( (QLEN, 19*19*18), dtype=np.float32)
# -- clientAddrs = list(range(QLEN))
# -- counter = 0
# -- #QLEN = QLEN
# -- print("Setting batch-size = %d" % QLEN)
# --
# -- def computeForward():
# --     global counter, inp, clientAddrs
# --     counter = 0
# --     myinp = inp.reshape( (QLEN, 18, 19, 19) )
# --     out = net(myinp).astype(np.float32)
# --
# --     ports = []      # never sent to an address twice each batch
# --     for i in range(QLEN):
# --         #print(out[0,0].data.tobytes())
# --         if not clientAddrs[i][1] in ports:
# --             sock.sendto(out[i, :].data, clientAddrs[i] )
# --             ports.append(clientAddrs[i][1])
# --         #self.transport.sendto(b"1234", self.ADDRS[i] )
# --
# -- while True:
# --     data, address = sock.recvfrom(19*19*18*4)
# --
# --     # if data:
# --     #     sent = sock.sendto(data, address)
# --     #     print('sent {} bytes back to {}'.format(
# --     #         sent, address))
# --
# --     if len(data) != 19*19*18*4:
# --         print("ERR")
# --
# --     inp[counter, :] = np.frombuffer(data, dtype=np.float32, count = 19*19*18)
# --     clientAddrs[counter] = address
# --
# --     # self.transport.sendto(data, addr)
# --     counter = counter + 1
# --     if counter == QLEN:
# --         computeForward()


inp = np.zeros( (QLEN, 19*19*18), dtype=np.float32)
ADDRS = list(range(QLEN))
counter = 0
print("\nBatch size = %d" % QLEN)

import threading
netlock = threading.Lock()
newNetWeight = None


class TCPServerClientProtocol(trollius.Protocol):
    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        print('Connection from {}'.format(peername))
        self.transport = transport

    def data_received(self, data):
        global counter, ADDRS, QLEN, inp, netlock, newNetWeight, net
        ADDRS[counter] = self.transport
        if len(data) != 19*19*18*4:
            print("ERR")
        inp[counter, :] = np.frombuffer(data, dtype=np.float32, count = 19*19*18)
        counter = counter + 1
        if counter == QLEN:
            counter = 0
            myinp = inp.reshape( (QLEN, 18, 19, 19) )
            netlock.acquire(True)   # BLOCK HERE
            if newNetWeight != None:
                del net
                gc.collect()  # hope that GPU memory is freed, not sure :-()
                weights, numBlocks, numFilters = newNetWeight
                print(" %d channels and %d blocks" % (numFilters, numBlocks) )
                net = LZN(weights, numBlocks, numFilters)
                print("...updated weight!")        
                newNetWeight = None
            netlock.release()
            out = net(myinp).astype(np.float32)
            for i in range(QLEN):
                try:
                    ADDRS[i].write(out[i, :].data)
                except Exception as ex:
                    print(ex)

loop = trollius.get_event_loop()
# Each client connection will create a new protocol instance
coro = loop.create_server(TCPServerClientProtocol, '127.0.0.1', 9999)
server = loop.run_until_complete(coro)

# Serve requests until Ctrl+C is pressed
print('\nServing on {}'.format(server.sockets[0].getsockname()))

import time
import gc

class MyWeightUpdater(threading.Thread):
    def run(self):
        global nethash, net, netlock, newNetWeight
        print("\nStarting a thread for auto updating lastest weights")             
        while True:
            newhash = getLastestNNHash()
            if newhash != nethash:
                txt = downloadBestNetworkWeight(newhash)
                print("New net arrived")
                nethash = newhash
                weights, numBlocks, numFilters = loadWeight(txt)
                netlock.acquire(True)  # BLOCK HERE
                newNetWeight = (weights, numBlocks, numFilters)
                netlock.release()
            time.sleep(10)                                      

worker = MyWeightUpdater(name = "Thread-1") 
worker.start()                                   

print("\nNow, you should run %d autogtp instances at different locations to start the playing process" % QLEN)
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

# Close the server
server.close()
loop.run_until_complete(server.wait_closed())
loop.close()


