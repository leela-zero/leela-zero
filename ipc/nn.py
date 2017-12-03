import sys
import glob
import gzip
import random
import math
import numpy as np
import trollius
from six.moves import urllib


if len(sys.argv) != 2 :
    print("Usage: %s batch-size" % sys.argv[0])
    sys.exit(-1)

QLEN     = int(sys.argv[1]) # alias: Batch size

print("Leela Zero TCP Neural Net Service")

def getLatestNNHash():
    txt = urllib.request.urlopen("http://zero.sjeng.org/best-network-hash").read().decode()
    net  = txt.split("\n")[0]
    return net

def downloadBestNetworkWeight(hash):
    try:
        return open(hash + ".txt", "r").read()
    except Exception as ex:
        txt = urllib.request.urlopen("http://zero.sjeng.org/networks/best-network").read().decode()
        f = open(hash + ".txt", "w")
        f.write(txt)
        f.close()
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

print("\nLoading latest network")
nethash = getLatestNNHash()
print("Hash: " + nethash)
print("Downloading weights")
txt     = downloadBestNetworkWeight(nethash)
print("Done!")

weights, numBlocks, numFilters = loadWeight(txt)
print(" %d channels and %d blocks" % (numFilters, numBlocks) )

import threading
netlock = threading.Lock()
newNetWeight = None

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
        global QLEN
        #f0 = T.tensor4(name + '_filter')
        w = np.asarray(loadW(), dtype=np.float32).reshape( (outc, inc, kernel_size, kernel_size) )
        #params.append(In(f0, value=w))
        f0 = shared(w)

        conv0 = conv2d(inp, f0, input_shape=(QLEN, inc, 19, 19),
                       border_mode='half',
                       filter_flip=False,
                       filter_shape=(outc, inc, kernel_size, kernel_size))
        b = loadW()  # zero bias
        if sum(abs(i) for i in b) != 0:
            print("ERROR! Should be 0")

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
    x = shared(np.zeros( (QLEN, 18, 19, 19), dtype=np.float32 ) ) # T.tensor4('input'))
    # params.append(x)
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
    return (x, function(params, out))

print("\nCompling the latest neural network")

from theano import *
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu
from theano.tensor.nnet.bn import batch_normalization_test as bn

net = LZN(weights, numBlocks, numFilters)

# inp = np.zeros( (1, 18, 19, 19), dtype=np.float32)
# for i in range(10000):
#     net[0].set_value(inp)
#     out = net[1]()
print("Done!")


import time
class MyWeightUpdater(threading.Thread):
    def run(self):
        global nethash, net, netlock, newNetWeight
        print("\nStarting a thread for auto updating latest weights\n")
        while True:
            newhash = getLatestNNHash()
            if newhash != nethash:
                txt = downloadBestNetworkWeight(newhash)
                print("New net arrived")
                nethash = newhash
                weights, numBlocks, numFilters = loadWeight(txt)
                netlock.acquire(True)  # BLOCK HERE
                newNetWeight = (weights, numBlocks, numFilters)
                netlock.release()
            time.sleep(10)

t2 = MyWeightUpdater(name = "Thread-1")
t2.daemon = True
t2.start()

